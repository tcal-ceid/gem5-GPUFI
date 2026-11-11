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

#include "gpu-compute/vector_register_file.hh"

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/GPUVRF.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/gpu_dyn_inst.hh"
#include "gpu-compute/register_file_cache.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/simple_pool_manager.hh"
#include "gpu-compute/wavefront.hh"
#include "params/VectorRegisterFile.hh"

namespace gem5
{

namespace
{

template <typename... Args>
void
propagationDetail(bool enabled, const char *format, Args&&... args)
{
    if (enabled) {
        inform(format, std::forward<Args>(args)...);
    }
}

struct VrfAnalysisState
{
    uint64_t totalAccesses = 0;
    uint64_t noEffectAccesses = 0;
    uint64_t effectiveActivations = 0;
    std::set<std::tuple<int, int, int, int>> accessedWavefronts;
    std::set<std::tuple<int, int>> accessedSimds;
    std::set<int> accessedCus;
    std::set<std::tuple<int, int, int, int, int>> accessedLaneContexts;
    std::set<int> accessedLaneIds;
    std::set<std::tuple<int, int, int, int>> effectiveWavefronts;
    std::set<std::tuple<int, int>> effectiveSimds;
    std::set<int> effectiveCus;
    std::set<std::tuple<int, int, int, int, int>> effectiveLaneContexts;
    std::set<int> effectiveLaneIds;

    uint64_t sourceActivations = 0;
    uint64_t registerWrites = 0;
    uint64_t vrfOriginWrites = 0;
    uint64_t srfOriginLaneWrites = 0;
    uint64_t mixedOriginWrites = 0;
    uint64_t merges = 0;
    uint64_t clears = 0;
    uint64_t reuseClears = 0;
    uint64_t multiRootWrites = 0;
    uint64_t sameLaneWrites = 0;
    uint64_t crossLaneWrites = 0;
    uint64_t broadcastSameWavefrontWrites = 0;
    uint64_t crossWavefrontWrites = 0;
    uint64_t crossCuWrites = 0;
    uint64_t unknownScopeWrites = 0;
    uint32_t maxDepth = 0;
    std::set<std::pair<uint64_t, int>> srfBroadcastDestinations;

    int faultCu = -1;
    int faultSimd = -1;
    int faultReg = -1;
    int faultLane = -1;
    int faultBit = -1;
    int stuckValue = -1;
};

VrfAnalysisState vrfAnalysis;

const char *
vrfExposureClass()
{
    if (vrfAnalysis.accessedWavefronts.empty()) {
        return "NO_ACCESS";
    }
    if (vrfAnalysis.accessedCus.size() > 1) {
        return "MULTI_CU";
    }
    if (vrfAnalysis.accessedWavefronts.size() > 1) {
        return "MULTI_WAVEFRONT_SINGLE_CU";
    }
    if (vrfAnalysis.accessedLaneContexts.size() > 1) {
        return "MULTI_LANE_SINGLE_WAVEFRONT";
    }
    return "SINGLE_LANE_SINGLE_WAVEFRONT";
}

} // anonymous namespace

VectorRegisterFile::VectorRegisterFile(const VectorRegisterFileParams &p)
    : RegisterFile(p)
{
    regFile.resize(numRegs());

    for (auto &reg : regFile) {
        reg.zero();
    }
}

void
VectorRegisterFile::setParent(ComputeUnit *_computeUnit)
{
    RegisterFile::setParent(_computeUnit);

    // Shader is not guaranteed to be connected while the ComputeUnit
    // constructor is still building its register files.
    // Propagation state will be initialized lazily during execution.
    vrfPropagationEvents.clear();
}

void
VectorRegisterFile::ensurePropagationState()
{
    if (!vrfPropagationEvents.empty()) {
        return;
    }

    if (!computeUnit ||
        !computeUnit->shader ||
        !computeUnit->shader->registerDepthTrackingEnabled()) {
        return;
    }

    vrfPropagationEvents.resize(numRegs());

    for (auto &registerState : vrfPropagationEvents) {
        registerState.fill(PropagationTag{});
    }
}

/**
 * Reads a source vector register across 64 execution lanes, evaluating active
 * permanent stuck-at faults and tracking per-lane causal provenance propagation.
 */
VectorRegisterFile::VecRegContainer&
VectorRegisterFile::readSource(
    int regIdx,
    Wavefront *wavefront,
    GPUDynInstPtr gpuDynInst)
{
    assert(wavefront);
    assert(gpuDynInst);

    ensurePropagationState();

    uint32_t *data = regFile.at(regIdx).as<uint32_t>();

    for (int lane = 0;
         lane < TheGpuISA::NumVecElemPerVecReg;
         ++lane) {

        const bool laneActive =
            wavefront->execMask(lane) ||
            gpuDynInst->ignoreExec();

        if (laneActive && propagationTrackingEnabled()) {
            const PropagationTag previousTag =
                propagationTag(regIdx, lane);

            if (previousTag.valid()) {
                const bool currentOwnerMatches =
                    previousTag.currentCu ==
                        computeUnit->cu_id &&
                    previousTag.currentSimd == simdId &&
                    previousTag.currentWfDynId ==
                        wavefront->wfDynId &&
                    previousTag.currentLane == lane;

                if (!currentOwnerMatches) {
                    const PropagationEventId clearEventId =
                        computeUnit->shader->
                            allocateVrfPropagationEventId();

                    assert(clearEventId != NoPropagationEvent);
                    vrfAnalysis.reuseClears++;

                    propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

                        "VRF_PROPAGATION event=TAINT_CLEAR "
                        "reason=REGISTER_REUSE origin_component=%s "
                        "event_id=%llu parent_event_id=%llu "
                        "root_activation_id=%llu depth=%u "
                        "tick=%llu physical_register=%d lane=%d "
                        "old_owner_cu=%d old_owner_simd=%d "
                        "old_owner_wf_dyn=%d old_owner_lane=%d "
                        "new_owner_cu=%d new_owner_simd=%d "
                        "new_owner_wf_dyn=%d new_owner_lane=%d",
                        registerPropagationOriginName(
                            previousTag.originComponent),
                        static_cast<unsigned long long>(
                            clearEventId),
                        static_cast<unsigned long long>(
                            previousTag.eventId),
                        static_cast<unsigned long long>(
                            previousTag.rootActivationId),
                        previousTag.depth,
                        static_cast<unsigned long long>(
                            curTick()),
                        regIdx,
                        lane,
                        previousTag.currentCu,
                        previousTag.currentSimd,
                        previousTag.currentWfDynId,
                        previousTag.currentLane,
                        computeUnit->cu_id,
                        simdId,
                        wavefront->wfDynId,
                        lane);

                    clearPropagationEvent(regIdx, lane);
                } else {
                    gpuDynInst->addVrfSourcePropagationTag(
                        lane, previousTag);
                }
            }
        }

        uint32_t currentValue = data[lane];

        for (const auto &faultEntry : hardFaults) {
            const HardKey &fault = faultEntry.first;
            const int stuckValue = faultEntry.second;

            if (fault.physReg != regIdx ||
                fault.lane != lane) {
                continue;
            }

            const uint32_t originalValue = currentValue;

            if (stuckValue == 0) {
                currentValue &=
                    ~(static_cast<uint32_t>(1U) << fault.bit);
            } else {
                currentValue |=
                    static_cast<uint32_t>(1U) << fault.bit;
            }

            if (!laneActive ||
                !computeUnit->shader->vrfDepthTrackingEnabled() ||
                originalValue == currentValue) {
                continue;
            }

            const PropagationEventId eventId =
                computeUnit->shader->
                    allocateVrfPropagationEventId();

            assert(eventId != NoPropagationEvent);
            vrfAnalysis.sourceActivations++;

            propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

                "VRF_PROPAGATION "
                "event=SOURCE_ACTIVATION "
                "origin_component=VRF destination_component=VRF "
                "event_id=%llu parent_event_id=0 "
                "fault_id=0 tick=%llu "
                "source_cu=%d source_simd=%d "
                "physical_register=%d lane=%d "
                "bit=%d stuck=%d "
                "original_value=0x%08x "
                "faulty_value=0x%08x "
                "wf_slot=%d wf_dyn=%d "
                "inst_seq=%llu pc=%#llx "
                "instruction=\"%s\"",
                static_cast<unsigned long long>(eventId),
                static_cast<unsigned long long>(curTick()),
                computeUnit->cu_id,
                simdId,
                regIdx,
                lane,
                fault.bit,
                stuckValue,
                originalValue,
                currentValue,
                wavefront->wfSlotId,
                wavefront->wfDynId,
                static_cast<unsigned long long>(
                    gpuDynInst->seqNum()),
                static_cast<unsigned long long>(
                    gpuDynInst->pc()),
                gpuDynInst->disassemble().c_str());

            PropagationTag activationTag;
            activationTag.eventId = eventId;
            activationTag.rootActivationId = eventId;
            activationTag.originComponent = RegisterPropagationOrigin::Vrf;
            activationTag.depth = 0;
            activationTag.originCu = computeUnit->cu_id;
            activationTag.originSimd = simdId;
            activationTag.originWfDynId =
                wavefront->wfDynId;
            activationTag.originLane = lane;

            activationTag.currentCu = computeUnit->cu_id;
            activationTag.currentSimd = simdId;
            activationTag.currentWfDynId =
                wavefront->wfDynId;
            activationTag.currentLane = lane;

            gpuDynInst->addVrfSourcePropagationTag(
                lane, activationTag);
        }

        // Applying the permanent fault changes the value returned by
        // this read, but a source read must not change the register's
        // stored provenance. Destination writes update provenance later.
        data[lane] = currentValue;
    }

    return regFile.at(regIdx);
}

/**
 * Records per-lane destination writes for the VRF, maintaining causal provenance chains.
 */
void
VectorRegisterFile::recordDestinationWrite(
    int regIdx,
    Wavefront *wavefront,
    GPUDynInstPtr gpuDynInst)
{
    assert(wavefront);
    assert(gpuDynInst);

    ensurePropagationState();

    if (!propagationTrackingEnabled()) {
        return;
    }

    const VectorMask &execMask = gpuDynInst->isLoad()
        ? gpuDynInst->exec_mask
        : wavefront->execMask();

    for (int lane = 0;
         lane < TheGpuISA::NumVecElemPerVecReg;
         ++lane) {

        if (!execMask[lane] && !gpuDynInst->ignoreExec()) {
            continue;
        }

        const PropagationTag &parentTag =
            gpuDynInst->vrfSourcePropagationTag(lane);

        const PropagationTag &previousTag =
            propagationTag(regIdx, lane);

        /*
         * A full destination write with no tainted VRF source is
         * considered a clean overwrite during this VRF-only phase.
         */
        if (!parentTag.valid()) {
            if (previousTag.valid()) {
                const PropagationEventId clearEventId =
                    computeUnit->shader->
                        allocateVrfPropagationEventId();

                assert(clearEventId != NoPropagationEvent);
                vrfAnalysis.clears++;

                propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

                    "VRF_PROPAGATION event=TAINT_CLEAR "
                    "reason=CLEAN_OVERWRITE origin_component=%s "
                    "event_id=%llu parent_event_id=%llu "
                    "root_activation_id=%llu "
                    "multiple_activation_roots=%d "
                    "previous_depth=%u "
                    "tick=%llu cu=%d simd=%d "
                    "destination_register=%d lane=%d "
                    "wf_slot=%d wf_dyn=%d "
                    "inst_seq=%llu pc=%#llx "
                    "instruction=\"%s\"",
                    registerPropagationOriginName(
                        previousTag.originComponent),
                    static_cast<unsigned long long>(
                        clearEventId),
                    static_cast<unsigned long long>(
                        previousTag.eventId),
                    static_cast<unsigned long long>(
                        previousTag.rootActivationId),
                    previousTag.multipleActivationRoots ? 1 : 0,
                    previousTag.depth,
                    static_cast<unsigned long long>(
                        curTick()),
                    computeUnit->cu_id,
                    simdId,
                    regIdx,
                    lane,
                    wavefront->wfSlotId,
                    wavefront->wfDynId,
                    static_cast<unsigned long long>(
                        gpuDynInst->seqNum()),
                    static_cast<unsigned long long>(
                        gpuDynInst->pc()),
                    gpuDynInst->disassemble().c_str());

                clearPropagationEvent(regIdx, lane);
            }

            continue;
        }

        const PropagationEventId writeEventId =
            computeUnit->shader->
                allocateVrfPropagationEventId();

        assert(writeEventId != NoPropagationEvent);

        PropagationTag destinationTag = parentTag;
        destinationTag.eventId = writeEventId;
        destinationTag.depth = parentTag.depth + 1;

        destinationTag.currentCu = computeUnit->cu_id;
        destinationTag.currentSimd = simdId;
        destinationTag.currentWfDynId =
            wavefront->wfDynId;
        destinationTag.currentLane = lane;

        setPropagationTag(
            regIdx,
            lane,
            destinationTag);

        vrfAnalysis.registerWrites++;
        vrfAnalysis.maxDepth = std::max(
            vrfAnalysis.maxDepth, destinationTag.depth);
        if (destinationTag.multipleActivationRoots) {
            vrfAnalysis.multiRootWrites++;
        }
        if (destinationTag.originComponent == RegisterPropagationOrigin::Vrf) {
            vrfAnalysis.vrfOriginWrites++;
        } else if (
            destinationTag.originComponent == RegisterPropagationOrigin::Srf) {
            vrfAnalysis.srfOriginLaneWrites++;
            vrfAnalysis.srfBroadcastDestinations.emplace(
                static_cast<uint64_t>(gpuDynInst->seqNum()), regIdx);
        } else if (
            destinationTag.originComponent == RegisterPropagationOrigin::Mixed) {
            vrfAnalysis.mixedOriginWrites++;
        }

        const char *scope = "UNKNOWN";

        if (destinationTag.originCu >= 0 &&
            destinationTag.originSimd >= 0 &&
            destinationTag.originWfDynId >= 0) {

            if (destinationTag.originLane < 0) {
                if (destinationTag.originCu == computeUnit->cu_id &&
                    destinationTag.originSimd == simdId &&
                    destinationTag.originWfDynId == wavefront->wfDynId) {
                    scope = "SRF_BROADCAST_SAME_WAVEFRONT";
                    vrfAnalysis.broadcastSameWavefrontWrites++;
                } else if (destinationTag.originCu == computeUnit->cu_id) {
                    scope = "CROSS_WAVEFRONT_SAME_CU";
                    vrfAnalysis.crossWavefrontWrites++;
                } else {
                    scope = "CROSS_CU";
                    vrfAnalysis.crossCuWrites++;
                }
            } else if (
                destinationTag.originCu == computeUnit->cu_id &&
                destinationTag.originSimd == simdId &&
                destinationTag.originWfDynId == wavefront->wfDynId &&
                destinationTag.originLane == lane) {
                scope = "SAME_LANE";
                vrfAnalysis.sameLaneWrites++;
            } else if (
                destinationTag.originCu == computeUnit->cu_id &&
                destinationTag.originSimd == simdId &&
                destinationTag.originWfDynId == wavefront->wfDynId) {
                scope = "CROSS_LANE_SAME_WAVEFRONT";
                vrfAnalysis.crossLaneWrites++;
            } else if (destinationTag.originCu == computeUnit->cu_id) {
                scope = "CROSS_WAVEFRONT_SAME_CU";
                vrfAnalysis.crossWavefrontWrites++;
            } else {
                scope = "CROSS_CU";
                vrfAnalysis.crossCuWrites++;
            }
        } else {
            vrfAnalysis.unknownScopeWrites++;
        }

        propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

            "VRF_PROPAGATION event=REGISTER_WRITE "
            "origin_component=%s destination_component=VRF "
            "event_id=%llu parent_event_id=%llu "
            "root_activation_id=%llu "
            "multiple_activation_roots=%d depth=%u "
            "scope=%s tick=%llu "
            "origin_cu=%d origin_simd=%d "
            "origin_wf_dyn=%d origin_lane=%d "
            "cu=%d simd=%d "
            "destination_register=%d lane=%d "
            "wf_slot=%d wf_dyn=%d "
            "inst_seq=%llu pc=%#llx "
            "instruction=\"%s\"",
            registerPropagationOriginName(
                destinationTag.originComponent),
            static_cast<unsigned long long>(
                writeEventId),
            static_cast<unsigned long long>(
                parentTag.eventId),
            static_cast<unsigned long long>(
                destinationTag.rootActivationId),
            destinationTag.multipleActivationRoots ? 1 : 0,
            destinationTag.depth,
            scope,
            static_cast<unsigned long long>(
                curTick()),
            destinationTag.originCu,
            destinationTag.originSimd,
            destinationTag.originWfDynId,
            destinationTag.originLane,
            computeUnit->cu_id,
            simdId,
            regIdx,
            lane,
            wavefront->wfSlotId,
            wavefront->wfDynId,
            static_cast<unsigned long long>(
                gpuDynInst->seqNum()),
            static_cast<unsigned long long>(
                gpuDynInst->pc()),
            gpuDynInst->disassemble().c_str());
    }
}

void
VectorRegisterFile::recordPropagationMerge(RegisterPropagationOrigin origin)
{
    vrfAnalysis.merges++;
}

void
VectorRegisterFile::dumpPropagationSummary() const
{
    if (!computeUnit || computeUnit->cu_id != 0 || simdId != 0) {
        return;
    }

    if (computeUnit->shader->vrfWidthTrackingEnabled()) {
        inform(
            "FAULT_ANALYSIS schema=1 category=WIDTH component=VRF "
            "fault_cu=%d fault_simd=%d physical_register=%d fault_lane=%d "
            "bit=%d stuck=%d total_accesses=%llu no_effect_accesses=%llu "
            "effective_activations=%llu accessed_unique_wavefronts=%llu "
            "accessed_unique_simds=%llu accessed_unique_cus=%llu "
            "accessed_unique_lane_contexts=%llu accessed_unique_lane_ids=%llu "
            "effective_unique_wavefronts=%llu effective_unique_simds=%llu "
            "effective_unique_cus=%llu effective_unique_lane_contexts=%llu "
            "effective_unique_lane_ids=%llu lane_semantics=PHYSICAL_VGPR_LANE "
            "exposure_class=%s",
            vrfAnalysis.faultCu,
            vrfAnalysis.faultSimd,
            vrfAnalysis.faultReg,
            vrfAnalysis.faultLane,
            vrfAnalysis.faultBit,
            vrfAnalysis.stuckValue,
            static_cast<unsigned long long>(vrfAnalysis.totalAccesses),
            static_cast<unsigned long long>(vrfAnalysis.noEffectAccesses),
            static_cast<unsigned long long>(vrfAnalysis.effectiveActivations),
            static_cast<unsigned long long>(
                vrfAnalysis.accessedWavefronts.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.accessedSimds.size()),
            static_cast<unsigned long long>(vrfAnalysis.accessedCus.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.accessedLaneContexts.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.accessedLaneIds.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.effectiveWavefronts.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.effectiveSimds.size()),
            static_cast<unsigned long long>(vrfAnalysis.effectiveCus.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.effectiveLaneContexts.size()),
            static_cast<unsigned long long>(
                vrfAnalysis.effectiveLaneIds.size()),
            vrfExposureClass());
    }

    if (computeUnit->shader->registerDepthTrackingEnabled()) {
        inform(
            "FAULT_ANALYSIS schema=1 category=DEPTH component=VRF "
            "source_activations=%llu register_writes=%llu "
            "vrf_origin_writes=%llu srf_origin_lane_writes=%llu "
            "srf_broadcast_destinations=%llu mixed_origin_writes=%llu "
            "merges=%llu clears=%llu reuse_clears=%llu "
            "multi_root_writes=%llu max_depth=%u same_lane_writes=%llu "
            "cross_lane_writes=%llu broadcast_same_wavefront_writes=%llu "
            "cross_wavefront_writes=%llu cross_cu_writes=%llu "
            "unknown_scope_writes=%llu",
            static_cast<unsigned long long>(vrfAnalysis.sourceActivations),
            static_cast<unsigned long long>(vrfAnalysis.registerWrites),
            static_cast<unsigned long long>(vrfAnalysis.vrfOriginWrites),
            static_cast<unsigned long long>(vrfAnalysis.srfOriginLaneWrites),
            static_cast<unsigned long long>(
                vrfAnalysis.srfBroadcastDestinations.size()),
            static_cast<unsigned long long>(vrfAnalysis.mixedOriginWrites),
            static_cast<unsigned long long>(vrfAnalysis.merges),
            static_cast<unsigned long long>(vrfAnalysis.clears),
            static_cast<unsigned long long>(vrfAnalysis.reuseClears),
            static_cast<unsigned long long>(vrfAnalysis.multiRootWrites),
            vrfAnalysis.maxDepth,
            static_cast<unsigned long long>(vrfAnalysis.sameLaneWrites),
            static_cast<unsigned long long>(vrfAnalysis.crossLaneWrites),
            static_cast<unsigned long long>(
                vrfAnalysis.broadcastSameWavefrontWrites),
            static_cast<unsigned long long>(
                vrfAnalysis.crossWavefrontWrites),
            static_cast<unsigned long long>(vrfAnalysis.crossCuWrites),
            static_cast<unsigned long long>(vrfAnalysis.unknownScopeWrites));
    }
}

bool
VectorRegisterFile::propagationTrackingEnabled() const
{
    return !vrfPropagationEvents.empty();
}

const VectorRegisterFile::PropagationTag&
VectorRegisterFile::propagationTag(int regIdx, int lane) const
{
    static const PropagationTag cleanTag;

    if (!propagationTrackingEnabled()) {
        return cleanTag;
    }

    return vrfPropagationEvents.at(regIdx).at(lane);
}

void
VectorRegisterFile::setPropagationTag(
    int regIdx,
    int lane,
    const PropagationTag &tag)
{
    if (!propagationTrackingEnabled()) {
        return;
    }

    assert(tag.valid());
    vrfPropagationEvents.at(regIdx).at(lane) = tag;
}

VectorRegisterFile::PropagationEventId
VectorRegisterFile::propagationEvent(int regIdx, int lane) const
{
    return propagationTag(regIdx, lane).eventId;
}

void
VectorRegisterFile::setPropagationEvent(
    int regIdx,
    int lane,
    PropagationEventId eventId)
{
    if (!propagationTrackingEnabled()) {
        return;
    }

    assert(eventId != NoPropagationEvent);

    PropagationTag tag;
    tag.eventId = eventId;

    setPropagationTag(regIdx, lane, tag);
}

void
VectorRegisterFile::clearPropagationEvent(int regIdx, int lane)
{
    if (!propagationTrackingEnabled()) {
        return;
    }

    vrfPropagationEvents.at(regIdx).at(lane) =
        PropagationTag{};
}

bool
VectorRegisterFile::operandsReady(Wavefront *w, GPUDynInstPtr ii) const
{
    bool src_ready = true, dst_ready=true;
    for (const auto& srcVecOp : ii->srcVecRegOperands()) {
        for (const auto& physIdx : srcVecOp.physIndices()) {
            if (regBusy(physIdx) &&
                    !computeUnit->rfc[simdId]->inRFC(physIdx)) {
                DPRINTF(GPUVRF, "RAW stall: WV[%d]: %s: physReg[%d]\n",
                        w->wfDynId, ii->disassemble(), physIdx);
                w->stats.numTimesBlockedDueRAWDependencies++;
                src_ready = false;
                break;
            }
        }
        if (!src_ready) {
            break;
        }
    }

    for (const auto& dstVecOp : ii->dstVecRegOperands()) {
        for (const auto& physIdx : dstVecOp.physIndices()) {
            if (regBusy(physIdx) &&
                    !computeUnit->rfc[simdId]->inRFC(physIdx)) {
                DPRINTF(GPUVRF, "WAX stall: WV[%d]: %s: physReg[%d]\n",
                        w->wfDynId, ii->disassemble(), physIdx);
                w->stats.numTimesBlockedDueWAXDependencies++;
                dst_ready = false;
                break;
            }
        }
        if (!dst_ready) {
            break;
        }
    }

    return src_ready && dst_ready;
}

void
VectorRegisterFile::scheduleWriteOperands(Wavefront *w, GPUDynInstPtr ii)
{
    for (const auto& dstVecOp : ii->dstVecRegOperands()) {
        for (const auto& physIdx : dstVecOp.physIndices()) {
            // If the instruction is atomic instruciton and the atomics do
            // not return value, then do not mark this reg as busy.
            if (!(ii->isAtomic() && !ii->isAtomicRet())) {
                /**
                 * if the instruction is a load with EXEC = 0, then we do not
                 * mark the reg. We do this to avoid a deadlock that can
                 * occur because a load reserves its destination regs before
                 * checking its exec mask, and in the cas it is 0, it will not
                 * send/recv any packets, and therefore it will never free its
                 * dst reg(s)
                 */
                if (ii->exec_mask.any()) {
                    markReg(physIdx, true);
                }
            }
        }
    }
}

void
VectorRegisterFile::waveExecuteInst(Wavefront *w, GPUDynInstPtr ii)
{
    // increment count of number of DWords read from VRF
    int DWords = ii->numSrcVecDWords();
    stats.registerReads += (DWords * w->execMask().count());


        // ------------------------------------------------------------
    if (computeUnit->shader->vrfWidthTrackingEnabled()) {
    // Permanent VRF fault: direct-consumer and scope tracing.
    //
    // We record only source-register reads that:
    //   1. access the faulted physical VGPR,
    //   2. include the faulted lane in the execution mask,
    //   3. are served by the VRF rather than the RFC,
    //   4. actually change the returned value.
    //
    // Campaign assumption: one permanent fault per simulation.
    // ------------------------------------------------------------
    auto &vrfActivatedConsumers = vrfAnalysis.effectiveLaneContexts;
    auto &vrfConsumerCus = vrfAnalysis.effectiveCus;
    auto &vrfConsumerWavefronts = vrfAnalysis.effectiveWavefronts;
    auto &vrfConsumerLanes = vrfAnalysis.effectiveLaneContexts;
    uint64_t &vrfActivationEvents = vrfAnalysis.effectiveActivations;
    static int vrfMaximumScope = 0;

    for (const auto& srcVecOp : ii->srcVecRegOperands()) {
        for (const auto& physIdx : srcVecOp.physIndices()) {

            // If the value is supplied by the register-file cache,
            // the physical VRF SRAM location is not directly read.
            if (computeUnit->rfc[simdId]->inRFC(physIdx)) {
                continue;
            }

            for (const auto& faultEntry : hardFaults) {
                const HardKey& fault = faultEntry.first;
                const int stuckValue = faultEntry.second;

                if (fault.physReg != physIdx) {
                    continue;
                }

                // The faulty lane must be active for this instruction.
                if (!w->execMask(fault.lane)) {
                    continue;
                }

                const uint32_t *registerData =
                    regFile[physIdx].as<const uint32_t>();

                const uint32_t observedValue =
                    registerData[fault.lane];

                uint32_t forcedValue = observedValue;

                if (stuckValue == 1) {
                    forcedValue |=
                        static_cast<uint32_t>(1U << fault.bit);
                } else {
                    forcedValue &=
                        static_cast<uint32_t>(~(1U << fault.bit));
                }

                const bool forcedNow =
                    observedValue != forcedValue;

                const auto consumerKey =
                    std::make_tuple(
                        computeUnit->cu_id,
                        simdId,
                        w->wfSlotId,
                        w->wfDynId,
                        fault.lane);

                vrfAnalysis.totalAccesses++;
                vrfAnalysis.accessedCus.insert(computeUnit->cu_id);
                vrfAnalysis.accessedSimds.emplace(
                    computeUnit->cu_id, simdId);
                vrfAnalysis.accessedWavefronts.emplace(
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId);
                vrfAnalysis.accessedLaneContexts.insert(consumerKey);
                vrfAnalysis.accessedLaneIds.insert(fault.lane);

                // The physical register was read, but the bit already
                // had the stuck-at value, so no numerical effect occurred.
                if (!forcedNow) {
                    // Keep the legacy direct-exposure accounting, but do
                    // not print the misleading post-execution message.
                    vrfAnalysis.noEffectAccesses++;
                    continue;
                }

                vrfActivationEvents++;
                vrfAnalysis.effectiveCus.insert(computeUnit->cu_id);
                vrfAnalysis.effectiveSimds.emplace(
                    computeUnit->cu_id, simdId);
                vrfAnalysis.effectiveWavefronts.emplace(
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId);
                vrfAnalysis.effectiveLaneIds.insert(fault.lane);

                // Print the first effective consumer for each
                // wavefront/lane/fault combination.
                if (!vrfActivatedConsumers.insert(
                        consumerKey).second) {
                    continue;
                }

                const int previousScope =
                    vrfMaximumScope;

                vrfConsumerCus.insert(
                    computeUnit->cu_id);

                vrfConsumerWavefronts.emplace(
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId);

                vrfConsumerLanes.emplace(
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId,
                    fault.lane);

                int newScope = 1; // LANE

                if (vrfConsumerCus.size() > 1) {
                    newScope = 4; // INTER_CU
                } else if (
                    vrfConsumerWavefronts.size() > 1) {
                    newScope = 3; // CU
                } else if (
                    vrfConsumerLanes.size() > 1) {
                    newScope = 2; // WAVEFRONT
                }

                if (newScope > vrfMaximumScope) {
                    vrfMaximumScope = newScope;
                }

                const char *scopeName = "NONE";

                switch (vrfMaximumScope) {
                  case 1:
                    scopeName = "LANE";
                    break;

                  case 2:
                    scopeName = "WAVEFRONT";
                    break;

                  case 3:
                    scopeName = "CU";
                    break;

                  case 4:
                    scopeName = "INTER_CU";
                    break;
                }

                const char *previousScopeName = "NONE";

                switch (previousScope) {
                  case 1:
                    previousScopeName = "LANE";
                    break;

                  case 2:
                    previousScopeName = "WAVEFRONT";
                    break;

                  case 3:
                    previousScopeName = "CU";
                    break;

                  case 4:
                    previousScopeName = "INTER_CU";
                    break;
                }

                propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

                    "FAULT_TRACE event=VRF_CONSUMER "
                    "fault_id=0 tick=%llu "
                    "physical_register=%d "
                    "bit=%d stuck=%d "
                    "destination_cu=%d "
                    "destination_simd=%d "
                    "destination_wf_slot=%d "
                    "destination_wf_dyn=%d "
                    "lane=%d "
                    "observed_value=0x%08x "
                    "forced_value=0x%08x "
                    "instruction=\"%s\"",
                    static_cast<unsigned long long>(
                        curTick()),
                    fault.physReg,
                    fault.bit,
                    stuckValue,
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId,
                    fault.lane,
                    observedValue,
                    forcedValue,
                    ii->disassemble().c_str());

                propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

                    "FAULT_TRACE event=VRF_SCOPE_STATE "
                    "fault_id=0 tick=%llu "
                    "previous_scope=%s "
                    "current_scope=%s "
                    "scope_changed=%d "
                    "activation_events=%llu "
                    "unique_cus=%llu "
                    "unique_wavefronts=%llu "
                    "unique_lanes=%llu "
                    "last_cu=%d "
                    "last_simd=%d "
                    "last_wf_slot=%d "
                    "last_wf_dyn=%d "
                    "last_lane=%d",
                    static_cast<unsigned long long>(
                        curTick()),
                    previousScopeName,
                    scopeName,
                    previousScope !=
                        vrfMaximumScope ? 1 : 0,
                    static_cast<unsigned long long>(
                        vrfActivationEvents),
                    static_cast<unsigned long long>(
                        vrfConsumerCus.size()),
                    static_cast<unsigned long long>(
                        vrfConsumerWavefronts.size()),
                    static_cast<unsigned long long>(
                        vrfConsumerLanes.size()),
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId,
                    fault.lane);
            }
        }
    }



    }

    for (const auto& dstVecOp : ii->dstVecRegOperands()) {
        for (const auto& physIdx : dstVecOp.physIndices()) {
            if (computeUnit->rfc[simdId]->inRFC(physIdx)) {
                stats.rfc_cache_write_hits += w->execMask().count();
            }
        }
    }

    for (const auto& srcVecOp : ii->srcVecRegOperands()) {
        for (const auto& physIdx : srcVecOp.physIndices()) {
            if (computeUnit->rfc[simdId]->inRFC(physIdx)) {
                stats.rfc_cache_read_hits += w->execMask().count();
            }
        }
    }

    uint64_t mask = w->execMask().to_ullong();
    int srams = w->execMask().size() / 4;
    for (int i = 0; i < srams; i++) {
        if (mask & 0xF) {
            stats.sramReads += DWords;
        }
        mask = mask >> 4;
    }

    if (!ii->isLoad()
        && !(ii->isAtomic() || ii->isMemSync())) {
        // TODO: compute proper delay
        // For now, it is based on largest operand size
        int opSize = ii->maxOperandSize();
        Cycles delay(opSize <= 4 ? computeUnit->spBypassLength()
            : computeUnit->dpBypassLength());
        Tick tickDelay = computeUnit->cyclesToTicks(delay);

        for (const auto& dstVecOp : ii->dstVecRegOperands()) {
            for (const auto& physIdx : dstVecOp.physIndices()) {
                enqRegFreeEvent(physIdx, tickDelay);
            }
        }
        // increment count of number of DWords written to VRF
        DWords = ii->numDstVecDWords();
        stats.registerWrites += (DWords * w->execMask().count());

        mask = w->execMask().to_ullong();
        srams = w->execMask().size() / 4;
        for (int i = 0; i < srams; i++) {
            if (mask & 0xF) {
                stats.sramWrites += DWords;
            }
            mask = mask >> 4;
        }
    }
}

void
VectorRegisterFile::scheduleWriteOperandsFromLoad(
    Wavefront *w, GPUDynInstPtr ii)
{
    assert(ii->isLoad() || ii->isAtomicRet());
    for (const auto& dstVecOp : ii->dstVecRegOperands()) {
        for (const auto& physIdx : dstVecOp.physIndices()) {
            enqRegFreeEvent(physIdx, computeUnit->clockPeriod());
        }
    }
    // increment count of number of DWords written to VRF
    int DWords = ii->numDstVecDWords();
    stats.registerWrites += (DWords * ii->exec_mask.count());

    uint64_t mask = ii->exec_mask.to_ullong();
    int srams = ii->exec_mask.size() / 4;
    for (int i = 0; i < srams; i++) {
        if (mask & 0xF) {
            stats.sramWrites += DWords;
        }
        mask = mask >> 4;
    }
}


/** Executes a transient single-bit XOR bit-flip on a specific VRF lane and bit. */
void
gem5::VectorRegisterFile::injectBitFlip(int physReg, int lane, int bit)
{
    // 1. Safety Check
    if (physReg < 0 || physReg >= regFile.size()) return;

    // 2. Perform the bit-flip
    auto &container = regFile.at(physReg);
    uint32_t *data = container.as<uint32_t>();
    uint32_t oldVal = data[lane];
    uint32_t newVal = oldVal ^ (1u << bit);
    data[lane] = newVal;


    // 4. Print the exact injection details
    inform("---------------------------------------------------------");
    inform("SINGLE VRF FAULT INJECTED AT TICK: %lu", curTick());
    inform("Location: CU%d | SIMD%d | Reg v%d | Lane %d | Bit %d",
            computeUnit->cu_id, simdId, physReg, lane, bit);
    inform("Effect  : 0x%08x -> 0x%08x", oldVal, newVal);

    if (oldVal == 0) {
        inform("Note    : Flipped a bit in a ZERO value (Value is now 0x%x)", newVal);
    }
    inform("---------------------------------------------------------");
}

/** Installs a permanent stuck-at constraint into the VRF hardFaults map. */
void
VectorRegisterFile::addHardFault(int physReg, int lane, int bit, int stuckValue)
{
    HardKey k{physReg, lane, bit};
    hardFaults[k] = (stuckValue != 0) ? 1 : 0;
    vrfAnalysis.faultCu = computeUnit->cu_id;
    vrfAnalysis.faultSimd = simdId;
    vrfAnalysis.faultReg = physReg;
    vrfAnalysis.faultLane = lane;
    vrfAnalysis.faultBit = bit;
    vrfAnalysis.stuckValue = stuckValue;

    inform("VRF HARD FAULT INSTALLED: Reg v%d Lane %d Bit %d stuck_at=%d",
           physReg, lane, bit, hardFaults[k]);
    if (computeUnit->shader->vrfWidthTrackingEnabled()) {
        propagationDetail(
                        computeUnit->shader->propagationVerboseEnabled(),

        "FAULT_TRACE event=VRF_SOURCE fault_id=0 "
        "tick=%llu component=VRF "
        "source_cu=%d source_simd=%d "
        "physical_register=%d source_lane=%d "
        "bit=%d stuck=%d",
        static_cast<unsigned long long>(curTick()),
        computeUnit->cu_id,
        simdId,
        physReg,
        lane,
        bit,
        hardFaults[k]);
    }

}

/** Evaluates and enforces permanent stuck-at constraints on vector register lane data. */
uint32_t
VectorRegisterFile::applyHardFaults(int physReg, int lane, uint32_t val) const
{
    // Apply all faults matching this (physReg, lane) across any bits
    // Since our map key includes bit, we scan bits 0..31 (32 checks). Minimal + fine.
    for (int b = 0; b < 32; ++b) {
        HardKey k{physReg, lane, b};
        auto it = hardFaults.find(k);
        if (it == hardFaults.end())
            continue;

        if (it->second == 0) {
            val &= ~(1u << b);   // force bit to 0
        } else {
            val |=  (1u << b);   // force bit to 1
        }
    }
    return val;
    }
} // namespace gem5
