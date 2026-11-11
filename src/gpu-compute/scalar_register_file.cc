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

#include "gpu-compute/scalar_register_file.hh"
#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <utility>


#include "base/logging.hh"
#include "debug/GPUSRF.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/gpu_dyn_inst.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/wavefront.hh"
#include "params/ScalarRegisterFile.hh"

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

struct SrfAnalysisState
{
    uint64_t totalAccesses = 0;
    uint64_t noEffectAccesses = 0;
    uint64_t effectiveActivations = 0;
    uint64_t activeLaneUnion = 0;
    uint64_t effectiveActiveLaneUnion = 0;
    uint64_t maxActiveLanes = 0;

    std::set<std::tuple<int, int, int, int>> accessedWavefronts;
    std::set<std::tuple<int, int>> accessedSimds;
    std::set<int> accessedCus;
    std::set<std::tuple<int, int, int, int>> effectiveWavefronts;
    std::set<std::tuple<int, int>> effectiveSimds;
    std::set<int> effectiveCus;

    uint64_t sourceActivations = 0;
    uint64_t registerWrites = 0;
    uint64_t merges = 0;
    uint64_t clears = 0;
    uint64_t reuseClears = 0;
    uint64_t multiRootWrites = 0;
    uint32_t maxDepth = 0;
    bool changedWavefront = false;
    bool changedCu = false;

    int faultCu = -1;
    int faultSimd = -1;
    int faultReg = -1;
    int faultBit = -1;
    int stuckValue = -1;
};

SrfAnalysisState srfAnalysis;

const char *
srfExposureClass()
{
    if (srfAnalysis.accessedWavefronts.empty()) {
        return "NO_ACCESS";
    }
    if (srfAnalysis.accessedCus.size() > 1) {
        return "MULTI_CU";
    }
    if (srfAnalysis.accessedSimds.size() > 1) {
        return "MULTI_SIMD_SINGLE_CU";
    }
    if (srfAnalysis.accessedWavefronts.size() > 1) {
        return "MULTI_WAVEFRONT_SINGLE_SIMD";
    }
    return "SINGLE_WAVEFRONT";
}

} // anonymous namespace

ScalarRegisterFile::ScalarRegisterFile(const ScalarRegisterFileParams &p)
    : RegisterFile(p)
{
    regFile.resize(numRegs(), 0);
}

void
ScalarRegisterFile::setParent(ComputeUnit *_computeUnit)
{
    RegisterFile::setParent(_computeUnit);
    srfPropagationEvents.clear();
}

void
ScalarRegisterFile::ensurePropagationState()
{
    if (!srfPropagationEvents.empty() || !computeUnit ||
        !computeUnit->shader->srfDepthTrackingEnabled()) {
        return;
    }

    srfPropagationEvents.assign(numRegs(), PropagationTag{});
}

bool
ScalarRegisterFile::propagationTrackingEnabled() const
{
    return !srfPropagationEvents.empty();
}

const ScalarRegisterFile::PropagationTag&
ScalarRegisterFile::propagationTag(int regIdx) const
{
    static const PropagationTag cleanTag;

    if (!propagationTrackingEnabled()) {
        return cleanTag;
    }

    return srfPropagationEvents.at(regIdx);
}

void
ScalarRegisterFile::setPropagationTag(
    int regIdx,
    const PropagationTag &tag)
{
    if (!propagationTrackingEnabled()) {
        return;
    }

    srfPropagationEvents.at(regIdx) = tag;
}

void
ScalarRegisterFile::clearPropagationTag(int regIdx)
{
    if (!propagationTrackingEnabled()) {
        return;
    }

    srfPropagationEvents.at(regIdx) = PropagationTag{};
}

/**
 * Reads a scalar register source operand, intercepting accesses to evaluate
 * active permanent stuck-at faults and tracking causal provenance propagation.
 */
ScalarRegisterFile::ScalarRegU32
ScalarRegisterFile::readSource(
    int regIdx,
    Wavefront *wavefront,
    GPUDynInstPtr gpuDynInst)
{
    assert(wavefront);
    assert(gpuDynInst);

    ensurePropagationState();

    if (propagationTrackingEnabled()) {
        const PropagationTag previousTag = propagationTag(regIdx);

        if (previousTag.valid()) {
            const bool currentOwnerMatches =
                previousTag.currentCu == computeUnit->cu_id &&
                previousTag.currentSimd == simdId &&
                previousTag.currentWfDynId == wavefront->wfDynId;

            if (!currentOwnerMatches) {
                const PropagationEventId clearEventId =
                    computeUnit->shader->
                        allocateRegisterPropagationEventId();

                assert(clearEventId != NoVrfPropagationEvent);
                srfAnalysis.reuseClears++;

                propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

                    "SRF_PROPAGATION event=TAINT_CLEAR "
                    "reason=REGISTER_REUSE origin_component=%s "
                    "event_id=%llu parent_event_id=%llu "
                    "root_activation_id=%llu depth=%u tick=%llu "
                    "physical_register=%d "
                    "old_owner_cu=%d old_owner_simd=%d "
                    "old_owner_wf_dyn=%d "
                    "new_owner_cu=%d new_owner_simd=%d "
                    "new_owner_wf_dyn=%d",
                    registerPropagationOriginName(
                        previousTag.originComponent),
                    static_cast<unsigned long long>(clearEventId),
                    static_cast<unsigned long long>(
                        previousTag.eventId),
                    static_cast<unsigned long long>(
                        previousTag.rootActivationId),
                    previousTag.depth,
                    static_cast<unsigned long long>(curTick()),
                    regIdx,
                    previousTag.currentCu,
                    previousTag.currentSimd,
                    previousTag.currentWfDynId,
                    computeUnit->cu_id,
                    simdId,
                    wavefront->wfDynId);

                clearPropagationTag(regIdx);
            } else {
                gpuDynInst->addSrfSourcePropagationTag(previousTag);
            }
        }
    }

    ScalarRegU32 currentValue = regFile.at(regIdx);
    const auto faultEntry = hardFaultMap.find(regIdx);

    if (faultEntry == hardFaultMap.end()) {
        return currentValue;
    }

    for (const auto &fault : faultEntry->second) {
        if (fault.bit < 0 || fault.bit >= 32) {
            continue;
        }

        const ScalarRegU32 originalValue = currentValue;

        if (fault.stuckValue == 1) {
            currentValue |= static_cast<ScalarRegU32>(1U << fault.bit);
        } else {
            currentValue &= static_cast<ScalarRegU32>(~(1U << fault.bit));
        }

        if (!computeUnit->shader->srfDepthTrackingEnabled() ||
            originalValue == currentValue) {
            continue;
        }

        const PropagationEventId eventId =
            computeUnit->shader->allocateRegisterPropagationEventId();

        assert(eventId != NoVrfPropagationEvent);
        srfAnalysis.sourceActivations++;

        propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

            "SRF_PROPAGATION event=SOURCE_ACTIVATION "
            "origin_component=SRF destination_component=SRF "
            "event_id=%llu parent_event_id=0 root_activation_id=%llu "
            "multiple_activation_roots=0 depth=0 "
            "fault_id=0 tick=%llu "
            "source_cu=%d source_simd=%d "
            "physical_register=%d bit=%d stuck=%d "
            "original_value=0x%08x faulty_value=0x%08x "
            "wf_slot=%d wf_dyn=%d active_lanes=%llu "
            "exec_mask=0x%016llx inst_seq=%llu pc=%#llx "
            "instruction=\"%s\"",
            static_cast<unsigned long long>(eventId),
            static_cast<unsigned long long>(eventId),
            static_cast<unsigned long long>(curTick()),
            computeUnit->cu_id,
            simdId,
            regIdx,
            fault.bit,
            fault.stuckValue,
            originalValue,
            currentValue,
            wavefront->wfSlotId,
            wavefront->wfDynId,
            static_cast<unsigned long long>(
                wavefront->execMask().count()),
            static_cast<unsigned long long>(
                wavefront->execMask().to_ullong()),
            static_cast<unsigned long long>(gpuDynInst->seqNum()),
            static_cast<unsigned long long>(gpuDynInst->pc()),
            gpuDynInst->disassemble().c_str());

        PropagationTag activationTag;
        activationTag.eventId = eventId;
        activationTag.rootActivationId = eventId;
        activationTag.originComponent = RegisterPropagationOrigin::Srf;
        activationTag.depth = 0;
        activationTag.originCu = computeUnit->cu_id;
        activationTag.originSimd = simdId;
        activationTag.originWfDynId = wavefront->wfDynId;
        activationTag.originLane = -1;
        activationTag.currentCu = computeUnit->cu_id;
        activationTag.currentSimd = simdId;
        activationTag.currentWfDynId = wavefront->wfDynId;
        activationTag.currentLane = -1;

        gpuDynInst->addSrfSourcePropagationTag(activationTag);
    }

    return currentValue;
}

/**
 * Records causal register writes for the SRF, updating provenance propagation chains.
 */
void
ScalarRegisterFile::recordDestinationWrite(
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

    const PropagationTag &parentTag =
        gpuDynInst->srfSourcePropagationTag();
    const PropagationTag previousTag = propagationTag(regIdx);

    if (!parentTag.valid()) {
        if (previousTag.valid()) {
            const PropagationEventId clearEventId =
                computeUnit->shader->
                    allocateRegisterPropagationEventId();

            assert(clearEventId != NoVrfPropagationEvent);
            srfAnalysis.clears++;

            propagationDetail(
                computeUnit->shader->propagationVerboseEnabled(),

                "SRF_PROPAGATION event=TAINT_CLEAR "
                "reason=CLEAN_OVERWRITE origin_component=%s "
                "event_id=%llu parent_event_id=%llu "
                "root_activation_id=%llu "
                "multiple_activation_roots=%d previous_depth=%u "
                "tick=%llu cu=%d simd=%d "
                "destination_register=%d wf_slot=%d wf_dyn=%d "
                "inst_seq=%llu pc=%#llx instruction=\"%s\"",
                registerPropagationOriginName(
                    previousTag.originComponent),
                static_cast<unsigned long long>(clearEventId),
                static_cast<unsigned long long>(previousTag.eventId),
                static_cast<unsigned long long>(
                    previousTag.rootActivationId),
                previousTag.multipleActivationRoots ? 1 : 0,
                previousTag.depth,
                static_cast<unsigned long long>(curTick()),
                computeUnit->cu_id,
                simdId,
                regIdx,
                wavefront->wfSlotId,
                wavefront->wfDynId,
                static_cast<unsigned long long>(gpuDynInst->seqNum()),
                static_cast<unsigned long long>(gpuDynInst->pc()),
                gpuDynInst->disassemble().c_str());

            clearPropagationTag(regIdx);
        }
        return;
    }

    const PropagationEventId writeEventId =
        computeUnit->shader->allocateRegisterPropagationEventId();

    assert(writeEventId != NoVrfPropagationEvent);

    PropagationTag destinationTag = parentTag;
    destinationTag.eventId = writeEventId;
    destinationTag.depth = parentTag.depth + 1;
    destinationTag.currentCu = computeUnit->cu_id;
    destinationTag.currentSimd = simdId;
    destinationTag.currentWfDynId = wavefront->wfDynId;
    destinationTag.currentLane = -1;

    setPropagationTag(regIdx, destinationTag);

    srfAnalysis.registerWrites++;
    srfAnalysis.maxDepth = std::max(
        srfAnalysis.maxDepth, destinationTag.depth);
    if (destinationTag.multipleActivationRoots) {
        srfAnalysis.multiRootWrites++;
    }

    const char *scope = "UNKNOWN";
    if (destinationTag.originCu >= 0 &&
        destinationTag.originWfDynId >= 0) {
        if (destinationTag.originCu == computeUnit->cu_id &&
            destinationTag.originSimd == simdId &&
            destinationTag.originWfDynId == wavefront->wfDynId) {
            scope = "SAME_WAVEFRONT";
        } else if (destinationTag.originCu == computeUnit->cu_id) {
            scope = "CROSS_WAVEFRONT_SAME_CU";
        } else {
            scope = "CROSS_CU";
        }
    }

    if (std::string(scope) == "CROSS_WAVEFRONT_SAME_CU") {
        srfAnalysis.changedWavefront = true;
    } else if (std::string(scope) == "CROSS_CU") {
        srfAnalysis.changedWavefront = true;
        srfAnalysis.changedCu = true;
    }

    propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

        "SRF_PROPAGATION event=REGISTER_WRITE "
        "origin_component=%s destination_component=SRF "
        "event_id=%llu parent_event_id=%llu "
        "root_activation_id=%llu multiple_activation_roots=%d "
        "depth=%u scope=%s tick=%llu "
        "origin_cu=%d origin_simd=%d origin_wf_dyn=%d "
        "cu=%d simd=%d destination_register=%d "
        "wf_slot=%d wf_dyn=%d inst_seq=%llu pc=%#llx "
        "instruction=\"%s\"",
        registerPropagationOriginName(destinationTag.originComponent),
        static_cast<unsigned long long>(writeEventId),
        static_cast<unsigned long long>(parentTag.eventId),
        static_cast<unsigned long long>(
            destinationTag.rootActivationId),
        destinationTag.multipleActivationRoots ? 1 : 0,
        destinationTag.depth,
        scope,
        static_cast<unsigned long long>(curTick()),
        destinationTag.originCu,
        destinationTag.originSimd,
        destinationTag.originWfDynId,
        computeUnit->cu_id,
        simdId,
        regIdx,
        wavefront->wfSlotId,
        wavefront->wfDynId,
        static_cast<unsigned long long>(gpuDynInst->seqNum()),
        static_cast<unsigned long long>(gpuDynInst->pc()),
        gpuDynInst->disassemble().c_str());
}

bool
ScalarRegisterFile::operandsReady(Wavefront *w, GPUDynInstPtr ii) const
{
    for (const auto& srcScalarOp : ii->srcScalarRegOperands()) {
        for (const auto& physIdx : srcScalarOp.physIndices()) {
            if (regBusy(physIdx)) {
                DPRINTF(GPUSRF, "RAW stall: WV[%d]: %s: physReg[%d]\n",
                        w->wfDynId, ii->disassemble(), physIdx);
                w->stats.numTimesBlockedDueRAWDependencies++;
                return false;
            }
        }
    }

    for (const auto& dstScalarOp : ii->dstScalarRegOperands()) {
        for (const auto& physIdx : dstScalarOp.physIndices()) {
            if (regBusy(physIdx)) {
                DPRINTF(GPUSRF, "WAX stall: WV[%d]: %s: physReg[%d]\n",
                        w->wfDynId, ii->disassemble(), physIdx);
                w->stats.numTimesBlockedDueWAXDependencies++;
                return false;
            }
        }
    }

    return true;
}

void
ScalarRegisterFile::scheduleWriteOperands(Wavefront *w, GPUDynInstPtr ii)
{
    for (const auto& dstScalarOp : ii->dstScalarRegOperands()) {
        for (const auto& physIdx : dstScalarOp.physIndices()) {
            // mark the destination scalar register as busy
            markReg(physIdx, true);
        }
    }
}

void
ScalarRegisterFile::waveExecuteInst(Wavefront *w, GPUDynInstPtr ii)
{
    stats.registerReads += ii->numSrcScalarDWords();

        // ------------------------------------------------------------
    if (computeUnit->shader->srfWidthTrackingEnabled()) {
    // Permanent SRF fault: direct-consumer and scope tracing.
    //
    // A scalar-register consumer is wavefront-scoped because
    // the scalar value is shared by the wavefront.
    //
    // Campaign assumption: one permanent fault per simulation.
    // ------------------------------------------------------------
    auto &srfActivatedConsumers = srfAnalysis.effectiveWavefronts;
    auto &srfConsumerCus = srfAnalysis.effectiveCus;
    auto &srfConsumerWavefronts = srfAnalysis.effectiveWavefronts;
    uint64_t &srfActivationEvents = srfAnalysis.effectiveActivations;
    static int srfMaximumScope = 0;

    for (const auto& srcScalarOp :
         ii->srcScalarRegOperands()) {

        for (const auto& physIdx :
             srcScalarOp.physIndices()) {

            const auto faultMapEntry =
                hardFaultMap.find(physIdx);

            if (faultMapEntry ==
                hardFaultMap.end()) {
                continue;
            }

            const ScalarRegU32 observedValue =
                regFile[physIdx];

            for (const auto& fault :
                 faultMapEntry->second) {

                ScalarRegU32 forcedValue =
                    observedValue;

                if (fault.stuckValue == 1) {
                    forcedValue |=
                        static_cast<ScalarRegU32>(
                            1U << fault.bit);
                } else {
                    forcedValue &=
                        static_cast<ScalarRegU32>(
                            ~(1U << fault.bit));
                }

                const bool forcedNow =
                    observedValue != forcedValue;

                const auto consumerKey =
                    std::make_tuple(
                        computeUnit->cu_id,
                        simdId,
                        w->wfSlotId,
                        w->wfDynId);

                const uint64_t execMask =
                    static_cast<uint64_t>(w->execMask().to_ullong());
                const uint64_t activeLanes =
                    static_cast<uint64_t>(w->execMask().count());

                srfAnalysis.totalAccesses++;
                srfAnalysis.accessedCus.insert(computeUnit->cu_id);
                srfAnalysis.accessedSimds.emplace(
                    computeUnit->cu_id, simdId);
                srfAnalysis.accessedWavefronts.insert(consumerKey);
                srfAnalysis.activeLaneUnion |= execMask;
                srfAnalysis.maxActiveLanes = std::max(
                    srfAnalysis.maxActiveLanes, activeLanes);

                if (!forcedNow) {
                    srfAnalysis.noEffectAccesses++;
                    static std::set<std::tuple<int, int, int, int>>
                        detailedNoEffectWavefronts;
                    if (detailedNoEffectWavefronts.insert(
                            consumerKey).second) {

                        propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

                            "FAULT_TRACE "
                            "event=SRF_READ_NO_EFFECT "
                            "fault_id=0 tick=%llu "
                            "source_cu=%d source_simd=%d "
                            "physical_register=%d "
                            "bit=%d stuck=%d "
                            "wf_slot=%d wf_dyn=%d "
                            "active_lanes=%llu "
                            "exec_mask=0x%016llx "
                            "observed_value=0x%08x "
                            "forced_value=0x%08x "
                            "instruction=\"%s\"",
                            static_cast<unsigned long long>(
                                curTick()),
                            computeUnit->cu_id,
                            simdId,
                            physIdx,
                            fault.bit,
                            fault.stuckValue,
                            w->wfSlotId,
                            w->wfDynId,
                            static_cast<unsigned long long>(
                                w->execMask().count()),
                            static_cast<unsigned long long>(
                                w->execMask().to_ullong()),
                            observedValue,
                            forcedValue,
                            ii->disassemble().c_str());
                    }

                    continue;
                }

                srfActivationEvents++;
                srfAnalysis.effectiveCus.insert(computeUnit->cu_id);
                srfAnalysis.effectiveSimds.emplace(
                    computeUnit->cu_id, simdId);
                srfAnalysis.effectiveActiveLaneUnion |= execMask;

                if (!srfActivatedConsumers.insert(
                        consumerKey).second) {
                    continue;
                }

                const int previousScope =
                    srfMaximumScope;

                srfConsumerCus.insert(
                    computeUnit->cu_id);

                srfConsumerWavefronts.emplace(
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId);

                int newScope = 1; // WAVEFRONT

                if (srfConsumerCus.size() > 1) {
                    newScope = 3; // INTER_CU
                } else if (
                    srfConsumerWavefronts.size() > 1) {
                    newScope = 2; // CU
                }

                if (newScope > srfMaximumScope) {
                    srfMaximumScope = newScope;
                }

                const char *scopeName = "NONE";

                switch (srfMaximumScope) {
                  case 1:
                    scopeName = "WAVEFRONT";
                    break;

                  case 2:
                    scopeName = "CU";
                    break;

                  case 3:
                    scopeName = "INTER_CU";
                    break;
                }

                const char *previousScopeName =
                    "NONE";

                switch (previousScope) {
                  case 1:
                    previousScopeName =
                        "WAVEFRONT";
                    break;

                  case 2:
                    previousScopeName =
                        "CU";
                    break;

                  case 3:
                    previousScopeName =
                        "INTER_CU";
                    break;
                }

                propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

                    "FAULT_TRACE event=SRF_CONSUMER "
                    "fault_id=0 tick=%llu "
                    "physical_register=%d "
                    "bit=%d stuck=%d "
                    "destination_cu=%d "
                    "destination_simd=%d "
                    "destination_wf_slot=%d "
                    "destination_wf_dyn=%d "
                    "active_lanes=%llu "
                    "exec_mask=0x%016llx "
                    "observed_value=0x%08x "
                    "forced_value=0x%08x "
                    "instruction=\"%s\"",
                    static_cast<unsigned long long>(
                        curTick()),
                    physIdx,
                    fault.bit,
                    fault.stuckValue,
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId,
                    static_cast<unsigned long long>(
                        w->execMask().count()),
                    static_cast<unsigned long long>(
                        w->execMask().to_ullong()),
                    observedValue,
                    forcedValue,
                    ii->disassemble().c_str());

                propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

                    "FAULT_TRACE event=SRF_SCOPE_STATE "
                    "fault_id=0 tick=%llu "
                    "previous_scope=%s "
                    "current_scope=%s "
                    "scope_changed=%d "
                    "activation_events=%llu "
                    "unique_cus=%llu "
                    "unique_wavefronts=%llu "
                    "last_cu=%d "
                    "last_simd=%d "
                    "last_wf_slot=%d "
                    "last_wf_dyn=%d "
                    "last_active_lanes=%llu",
                    static_cast<unsigned long long>(
                        curTick()),
                    previousScopeName,
                    scopeName,
                    previousScope !=
                        srfMaximumScope ? 1 : 0,
                    static_cast<unsigned long long>(
                        srfActivationEvents),
                    static_cast<unsigned long long>(
                        srfConsumerCus.size()),
                    static_cast<unsigned long long>(
                        srfConsumerWavefronts.size()),
                    computeUnit->cu_id,
                    simdId,
                    w->wfSlotId,
                    w->wfDynId,
                    static_cast<unsigned long long>(
                        w->execMask().count()));
            }
        }
    }


    }

    if (!ii->isLoad() && !(ii->isAtomic() || ii->isMemSync())) {
        Cycles delay(computeUnit->scalarPipeLength());
        Tick tickDelay = computeUnit->cyclesToTicks(delay);

        for (const auto& dstScalarOp : ii->dstScalarRegOperands()) {
            for (const auto& physIdx : dstScalarOp.physIndices()) {
                enqRegFreeEvent(physIdx, tickDelay);
            }
        }

        stats.registerWrites += ii->numDstScalarDWords();
    }
}

void
ScalarRegisterFile::scheduleWriteOperandsFromLoad(Wavefront *w,
                                                  GPUDynInstPtr ii)
{
    assert(ii->isLoad() || ii->isAtomicRet());
    for (const auto& dstScalarOp : ii->dstScalarRegOperands()) {
        for (const auto& physIdx : dstScalarOp.physIndices()) {
            enqRegFreeEvent(physIdx, computeUnit->clockPeriod());
        }
    }

    stats.registerWrites += ii->numDstScalarDWords();
}



void
ScalarRegisterFile::recordPropagationMerge()
{
    srfAnalysis.merges++;
}

void
ScalarRegisterFile::dumpPropagationSummary() const
{
    if (!computeUnit || computeUnit->cu_id != 0 || simdId != 0) {
        return;
    }

    if (computeUnit->shader->srfWidthTrackingEnabled()) {
        inform(
            "FAULT_ANALYSIS schema=1 category=WIDTH component=SRF "
            "fault_cu=%d fault_simd=%d physical_register=%d bit=%d "
            "stuck=%d total_accesses=%llu no_effect_accesses=%llu "
            "effective_activations=%llu accessed_unique_wavefronts=%llu "
            "accessed_unique_simds=%llu accessed_unique_cus=%llu "
            "effective_unique_wavefronts=%llu effective_unique_simds=%llu "
            "effective_unique_cus=%llu active_lane_union_count=%u "
            "effective_active_lane_union_count=%u max_active_lanes=%llu "
            "lane_semantics=SCALAR_BROADCAST exposure_class=%s",
            srfAnalysis.faultCu,
            srfAnalysis.faultSimd,
            srfAnalysis.faultReg,
            srfAnalysis.faultBit,
            srfAnalysis.stuckValue,
            static_cast<unsigned long long>(srfAnalysis.totalAccesses),
            static_cast<unsigned long long>(srfAnalysis.noEffectAccesses),
            static_cast<unsigned long long>(srfAnalysis.effectiveActivations),
            static_cast<unsigned long long>(
                srfAnalysis.accessedWavefronts.size()),
            static_cast<unsigned long long>(
                srfAnalysis.accessedSimds.size()),
            static_cast<unsigned long long>(srfAnalysis.accessedCus.size()),
            static_cast<unsigned long long>(
                srfAnalysis.effectiveWavefronts.size()),
            static_cast<unsigned long long>(
                srfAnalysis.effectiveSimds.size()),
            static_cast<unsigned long long>(srfAnalysis.effectiveCus.size()),
            static_cast<unsigned>(__builtin_popcountll(
                srfAnalysis.activeLaneUnion)),
            static_cast<unsigned>(__builtin_popcountll(
                srfAnalysis.effectiveActiveLaneUnion)),
            static_cast<unsigned long long>(srfAnalysis.maxActiveLanes),
            srfExposureClass());
    }

    if (computeUnit->shader->srfDepthTrackingEnabled()) {
        inform(
            "FAULT_ANALYSIS schema=1 category=DEPTH component=SRF "
            "source_activations=%llu register_writes=%llu merges=%llu "
            "clears=%llu reuse_clears=%llu multi_root_writes=%llu "
            "max_depth=%u changed_wavefront=%d changed_cu=%d",
            static_cast<unsigned long long>(srfAnalysis.sourceActivations),
            static_cast<unsigned long long>(srfAnalysis.registerWrites),
            static_cast<unsigned long long>(srfAnalysis.merges),
            static_cast<unsigned long long>(srfAnalysis.clears),
            static_cast<unsigned long long>(srfAnalysis.reuseClears),
            static_cast<unsigned long long>(srfAnalysis.multiRootWrites),
            srfAnalysis.maxDepth,
            srfAnalysis.changedWavefront ? 1 : 0,
            srfAnalysis.changedCu ? 1 : 0);
    }
}

/** Executes a transient single-bit XOR bit-flip on the targeted SRF register. */
void
ScalarRegisterFile::injectBitFlip(int physReg, int bit)
{
    if (physReg >= regFile.size() || bit >= 32)
        return;

    uint32_t &reg = regFile[physReg];

    uint32_t before = reg;
    reg ^= (1u << bit);
    uint32_t after = reg;

    inform("SRF TRANSIENT INJECTED | Reg=%d Bit=%d | BEFORE=0x%08x AFTER=0x%08x",
           physReg, bit, before, after);
}

/** Installs a permanent stuck-at fault constraint into the SRF hardFaultMap. */
void
ScalarRegisterFile::addHardFault(int physReg, int bit, int stuckValue)
{
    hardFaultMap[physReg].push_back({bit, stuckValue});
    srfAnalysis.faultCu = computeUnit->cu_id;
    srfAnalysis.faultSimd = simdId;
    srfAnalysis.faultReg = physReg;
    srfAnalysis.faultBit = bit;
    srfAnalysis.stuckValue = stuckValue;

    inform("SRF HARD FAULT INSTALLED | Reg=%d Bit=%d Stuck=%d",
           physReg, bit, stuckValue);

    if (computeUnit->shader->srfWidthTrackingEnabled()) {
        propagationDetail(
                    computeUnit->shader->propagationVerboseEnabled(),

        "FAULT_TRACE event=SRF_SOURCE fault_id=0 "
        "tick=%llu component=SRF "
        "source_cu=%d source_simd=%d "
        "physical_register=%d bit=%d stuck=%d",
        static_cast<unsigned long long>(curTick()),
        computeUnit->cu_id,
        simdId,
        physReg,
        bit,
        stuckValue);
    }

}

/** Evaluates and enforces permanent stuck-at constraints on a scalar register value. */
ScalarRegisterFile::ScalarRegU32
ScalarRegisterFile::applyHardFaults(int physReg, ScalarRegU32 value) const
{
    auto it = hardFaultMap.find(physReg);
    if (it == hardFaultMap.end())
        return value;

    for (const auto &hf : it->second) {
        if (hf.bit < 0 || hf.bit >= 32)
            continue;

        if (hf.stuckValue)
            value |= (1u << hf.bit);
        else
            value &= ~(1u << hf.bit);
    }



    return value;
}


} // namespace gem5
