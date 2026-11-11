/*
 * Copyright (c) 2015-2017 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __VECTOR_REGISTER_FILE_HH__
#define __VECTOR_REGISTER_FILE_HH__

#include <array>
#include <cstdint>
#include <set>
#include <tuple>

#include "arch/gpu_isa.hh"
#include "config/the_gpu_isa.hh"
#include "debug/GPUTrace.hh"
#include "debug/GPUVRF.hh"
#include "gpu-compute/register_file.hh"
#include "gpu-compute/vrf_propagation.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{
#include <unordered_map>

/** Key structure identifying a specific physical register, lane, and bit for permanent VRF faults. */
struct HardKey {
    int physReg;
    int lane;
    int bit;
    bool operator==(const HardKey& o) const {
        return physReg == o.physReg && lane == o.lane && bit == o.bit;
    }
};

/** Hash functor for HardKey to support unordered_map storage. */
struct HardKeyHash {
    std::size_t operator()(const HardKey& k) const noexcept {
        // simple hash combine
        return ((std::size_t)k.physReg * 1315423911u) ^
               ((std::size_t)k.lane    * 2654435761u) ^
               ((std::size_t)k.bit     * 97531u);
    }
};



// Vector Register File
class VectorRegisterFile : public RegisterFile
{
  public:




    void injectBitFlip(int physReg, int lane, int bit);

    void addHardFault(int physReg, int lane, int bit, int stuckValue);

   using VecRegContainer = TheGpuISA::VecRegContainerU32;

    using PropagationEventId = VrfPropagationEventId;
    using PropagationTag = VrfPropagationTag;

    static constexpr PropagationEventId NoPropagationEvent =
        NoVrfPropagationEvent;

    bool propagationTrackingEnabled() const;

    const PropagationTag&
    propagationTag(int regIdx, int lane) const;

    void
    setPropagationTag(
        int regIdx,
        int lane,
        const PropagationTag &tag);

    /*
     * Compatibility wrappers used by the current implementation.
     * Later steps will migrate source activation and writes to full tags.
     */
    PropagationEventId
    propagationEvent(int regIdx, int lane) const;

    void
    setPropagationEvent(
        int regIdx,
        int lane,
        PropagationEventId eventId);

    void
    clearPropagationEvent(int regIdx, int lane);

    /**
     * Read a source VGPR while capturing fault activation and provenance.
     */
    VecRegContainer&
    readSource(
        int regIdx,
        Wavefront *wavefront,
        GPUDynInstPtr gpuDynInst);

    /**
     * Update per-lane provenance after an instruction writes a VGPR.
     */
    void
    recordDestinationWrite(
        int regIdx,
        Wavefront *wavefront,
        GPUDynInstPtr gpuDynInst);

    void recordPropagationMerge(RegisterPropagationOrigin origin);
    void dumpPropagationSummary() const;

    VectorRegisterFile(const VectorRegisterFileParams &p);
    ~VectorRegisterFile() { }

    virtual bool operandsReady(Wavefront *w, GPUDynInstPtr ii) const override;
    virtual void scheduleWriteOperands(Wavefront *w,
                                       GPUDynInstPtr ii) override;
    virtual void scheduleWriteOperandsFromLoad(Wavefront *w,
                                               GPUDynInstPtr ii) override;
    virtual void waveExecuteInst(Wavefront *w, GPUDynInstPtr ii) override;

    void
    setParent(ComputeUnit *_computeUnit) override;

    // Read a register that is writeable (e.g., a DST operand)
    VecRegContainer&
readWriteable(int regIdx)
{
    uint32_t *data = regFile[regIdx].as<uint32_t>();
    for (int lane = 0; lane < 64; ++lane) {
        data[lane] = applyHardFaults(regIdx, lane, data[lane]);
    }
    return regFile[regIdx];
}

    // Read a register that is not writeable (e.g., src operand)
    VecRegContainer
read(int regIdx) const
{
    // Create a copy to hold the corrupted data
    VecRegContainer corruptedCopy = regFile[regIdx];
    uint32_t *data = corruptedCopy.as<uint32_t>();

    // Apply faults to all 64 lanes
    for (int lane = 0; lane < 64; ++lane) {
        data[lane] = applyHardFaults(regIdx, lane, data[lane]);
    }
    return corruptedCopy;
}

    // Write a register
    void
write(int regIdx, const VecRegContainer &value)
{
    VecRegContainer out = value;

    const uint32_t *src = value.as<const uint32_t>();
    uint32_t *dst = out.as<uint32_t>();

    // 64 lanes (wavefront size)
    for (int lane = 0; lane < 64; ++lane) {
        dst[lane] = applyHardFaults(regIdx, lane, src[lane]);
    }

    regFile[regIdx] = out;
}


    void
    printReg(Wavefront *wf, int regIdx) const
    {
#ifndef NDEBUG
        const auto &vec_reg_cont = regFile[regIdx];
        auto vgpr = vec_reg_cont.as<TheGpuISA::VecElemU32>();

        for (int lane = 0; lane < TheGpuISA::NumVecElemPerVecReg; ++lane) {
            if (wf->execMask(lane)) {
                DPRINTF(GPUVRF, "WF[%d][%d]: WV[%d] v[%d][%d] = %#x\n",
                    wf->simdId, wf->wfSlotId, wf->wfDynId, regIdx, lane,
                    vgpr[lane]);
                DPRINTF(GPUTrace, "WF[%d][%d]: WV[%d] v[%d][%d] = %#x (%f)\n",
                    wf->simdId, wf->wfSlotId, wf->wfDynId, regIdx, lane,
                    vgpr[lane], *reinterpret_cast<const float*>(&vgpr[lane]));
            }
        }
#endif
    }

   private:
    using LanePropagationState = std::array<
        PropagationTag,
        TheGpuISA::NumVecElemPerVecReg>;

    std::vector<VecRegContainer> regFile;

    // Empty when propagation tracking is disabled.
    // Otherwise indexed as [physical VGPR][lane].
    std::vector<LanePropagationState> vrfPropagationEvents;

    void ensurePropagationState();

    std::unordered_map<HardKey, int, HardKeyHash> hardFaults;

    uint32_t applyHardFaults(int physReg, int lane, uint32_t val) const;
};


} // namespace gem5





#endif // __VECTOR_REGISTER_FILE_HH__
