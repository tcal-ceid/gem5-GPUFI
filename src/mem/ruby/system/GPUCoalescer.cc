/*
 * Copyright (c) 2013-2015 Advanced Micro Devices, Inc.
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

#include "mem/ruby/system/GPUCoalescer.hh"
#include "sim/sim_exit.hh"

#include <set>
#include <tuple>

#include "base/compiler.hh"
#include "base/logging.hh"
#include "base/str.hh"
#include "cpu/testers/rubytest/RubyTester.hh"
#include "debug/GPUCoalescer.hh"
#include "debug/MemoryAccess.hh"
#include "debug/ProtocolTrace.hh"
#include "debug/RubyHitMiss.hh"
#include "debug/RubyPort.hh"
#include "debug/RubyStats.hh"
#include "gpu-compute/shader.hh"
#include "mem/packet.hh"
#include "mem/ruby/common/DataBlock.hh"
#include "mem/ruby/common/SubBlock.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/profiler/Profiler.hh"
#include "mem/ruby/slicc_interface/AbstractController.hh"
#include "mem/ruby/slicc_interface/RubyRequest.hh"
#include "mem/ruby/structures/CacheMemory.hh"
#include "mem/ruby/system/RubySystem.hh"
#include "params/RubyGPUCoalescer.hh"

namespace gem5
{

namespace ruby
{

namespace
{

// Compact direct-exposure width tracking for permanent TCP faults.
// This deliberately does not propagate values after the cache consumer.
struct TcpWidthState
{
    uint64_t totalAccesses = 0;
    uint64_t noEffectAccesses = 0;
    uint64_t effectiveActivations = 0;

    std::set<int> accessedCus;
    std::set<std::tuple<int, int>> accessedSimds;
    std::set<std::tuple<int, int, int, int>> accessedWavefronts;
    std::set<std::tuple<int, int, int, int, int>> accessedLaneContexts;
    std::set<int> accessedLaneIndices;

    std::set<int> effectiveCus;
    std::set<std::tuple<int, int>> effectiveSimds;
    std::set<std::tuple<int, int, int, int>> effectiveWavefronts;
    std::set<std::tuple<int, int, int, int, int>> effectiveLaneContexts;
    std::set<int> effectiveLaneIndices;

    int faultCache = -1;
    int faultSet = -1;
    int faultWay = -1;
    int faultByte = -1;
    int faultBit = -1;
    int stuckValue = -1;

    bool summaryPrinted = false;
};

TcpWidthState tcpWidth;

/**
 * Classifies the hierarchical execution fan-out scope across CUs, SIMDs,
 * wavefronts, and execution lanes for TCP fault accesses.
 */
const char *
tcpExposureClass(
    const std::set<int>& cus,
    const std::set<std::tuple<int, int>>& simds,
    const std::set<std::tuple<int, int, int, int>>& wavefronts,
    const std::set<std::tuple<int, int, int, int, int>>& lanes)
{
    if (lanes.empty()) {
        return "NO_ACCESS";
    }
    if (cus.size() > 1) {
        return "MULTI_CU";
    }
    if (simds.size() > 1) {
        return "MULTI_SIMD_SINGLE_CU";
    }
    if (wavefronts.size() > 1) {
        return "MULTI_WAVEFRONT_SINGLE_SIMD";
    }
    if (lanes.size() > 1) {
        return "MULTI_LANE_SINGLE_WAVEFRONT";
    }
    return "SINGLE_LANE";
}

/**
 * Records a direct TCP cache consumer access, distinguishing between dormant
 * (no-effect) reads and effective fault activations.
 */
void
recordTcpWidth(
    int cache,
    const CacheMemory::TcpHardFaultHit& hit,
    int cu,
    int simd,
    int wfSlot,
    int wfDyn,
    int lane)
{
    tcpWidth.totalAccesses++;
    tcpWidth.faultCache = cache;
    tcpWidth.faultSet = hit.set;
    tcpWidth.faultWay = hit.way;
    tcpWidth.faultByte = hit.byte;
    tcpWidth.faultBit = hit.bit;
    tcpWidth.stuckValue = hit.stuck_val;

    tcpWidth.accessedCus.insert(cu);
    tcpWidth.accessedSimds.emplace(cu, simd);
    tcpWidth.accessedWavefronts.emplace(cu, simd, wfSlot, wfDyn);
    tcpWidth.accessedLaneContexts.emplace(cu, simd, wfSlot, wfDyn, lane);
    tcpWidth.accessedLaneIndices.insert(lane);

    if (!hit.forced_now) {
        tcpWidth.noEffectAccesses++;
        return;
    }

    tcpWidth.effectiveActivations++;
    tcpWidth.effectiveCus.insert(cu);
    tcpWidth.effectiveSimds.emplace(cu, simd);
    tcpWidth.effectiveWavefronts.emplace(cu, simd, wfSlot, wfDyn);
    tcpWidth.effectiveLaneContexts.emplace(cu, simd, wfSlot, wfDyn, lane);
    tcpWidth.effectiveLaneIndices.insert(lane);
}

} // anonymous namespace



UncoalescedTable::UncoalescedTable(GPUCoalescer *gc)
    : coalescer(gc)
{
}

void
UncoalescedTable::insertPacket(PacketPtr pkt)
{
    uint64_t seqNum = pkt->req->getReqInstSeqNum();

    instMap[seqNum].push_back(pkt);
    DPRINTF(GPUCoalescer, "Adding 0x%X seqNum %d to map. (map %d vec %d)\n",
            pkt->getAddr(), seqNum, instMap.size(), instMap[seqNum].size());
}

void
UncoalescedTable::insertReqType(PacketPtr pkt, RubyRequestType type)
{
    uint64_t seqNum = pkt->req->getReqInstSeqNum();

    reqTypeMap[seqNum] = type;
}

bool
UncoalescedTable::packetAvailable()
{
    return !instMap.empty();
}

void
UncoalescedTable::initPacketsRemaining(InstSeqNum seqNum, int count)
{
    if (!instPktsRemaining.count(seqNum)) {
        instPktsRemaining[seqNum] = count;
    }
}

int
UncoalescedTable::getPacketsRemaining(InstSeqNum seqNum)
{
    return instPktsRemaining[seqNum];
}

void
UncoalescedTable::setPacketsRemaining(InstSeqNum seqNum, int count)
{
    instPktsRemaining[seqNum] = count;
}

PerInstPackets*
UncoalescedTable::getInstPackets(int offset)
{
    if (offset >= instMap.size()) {
        return nullptr;
    }

    auto instMapIter = instMap.begin();
    std::advance(instMapIter, offset);

    return &(instMapIter->second);
}

void
UncoalescedTable::updateResources()
{
    for (auto iter = instMap.begin(); iter != instMap.end(); ) {
        InstSeqNum seq_num = iter->first;
        DPRINTF(GPUCoalescer, "%s checking remaining pkts for %d\n",
                coalescer->name().c_str(), seq_num);
        assert(instPktsRemaining.count(seq_num));

        if (instPktsRemaining[seq_num] == 0) {
            assert(iter->second.empty());

            // Remove from both maps
            instMap.erase(iter++);
            instPktsRemaining.erase(seq_num);

            // Release the token if the Ruby system is not in cooldown
            // or warmup phases. When in these phases, the RubyPorts
            // are accessed directly using the makeRequest() command
            // instead of accessing through the port. This makes
            // sending tokens through the port unnecessary
            if (!coalescer->getRubySystem()->getWarmupEnabled() &&
                !coalescer->getRubySystem()->getCooldownEnabled()) {
                if (reqTypeMap[seq_num] != RubyRequestType_FLUSH) {
                    DPRINTF(GPUCoalescer,
                            "Returning token seqNum %d\n", seq_num);
                    coalescer->getGMTokenPort().sendTokens(1);
                }
            }

            reqTypeMap.erase(seq_num);
        } else {
            ++iter;
        }
    }
}

bool
UncoalescedTable::areRequestsDone(const uint64_t instSeqNum) {
    // iterate the instructions held in UncoalescedTable to see whether there
    // are more requests to issue; if yes, not yet done; otherwise, done
    for (auto& inst : instMap) {
        DPRINTF(GPUCoalescer, "instSeqNum= %d, pending packets=%d\n"
            ,inst.first, inst.second.size());
        if (inst.first == instSeqNum) { return false; }
    }

    return true;
}

void
UncoalescedTable::printRequestTable(std::stringstream& ss)
{
    ss << "Listing pending packets from " << instMap.size() << " instructions";

    for (auto& inst : instMap) {
        ss << "\tAddr: " << coalescer->printAddress(inst.first) << " with "
           << inst.second.size() << " pending packets" << std::endl;
    }
}

void
UncoalescedTable::checkDeadlock(Tick threshold)
{
    Tick current_time = curTick();

    for (auto &it : instMap) {
        for (auto &pkt : it.second) {
            if (current_time - pkt->req->time() > threshold) {
                std::stringstream ss;
                printRequestTable(ss);

                panic("Possible Deadlock detected. Aborting!\n"
                     "version: %d request.paddr: 0x%x uncoalescedTable: %d "
                     "current time: %u issue_time: %d difference: %d\n"
                     "Request Tables:\n\n%s", coalescer->getId(),
                      pkt->getAddr(), instMap.size(), current_time,
                      pkt->req->time(), current_time - pkt->req->time(),
                      ss.str());
            }
        }
    }
}

GPUCoalescer::GPUCoalescer(const Params &p)
    : RubyPort(p),
      issueEvent([this]{
    registerExitCallback([this] { exitCallback(); }); completeIssue(); }, "Issue coalesced request",
                 false, Event::Progress_Event_Pri),
      uncoalescedTable(this),
      deadlockCheckEvent([this]{ wakeup(); }, "GPUCoalescer deadlock check"),
      stats(this),
      gmTokenPort(name() + ".gmTokenPort")
{
    m_store_waiting_on_load_cycles = 0;
    m_store_waiting_on_store_cycles = 0;
    m_load_waiting_on_store_cycles = 0;
    m_load_waiting_on_load_cycles = 0;

    m_outstanding_count = 0;

    coalescingWindow = p.max_coalesces_per_cycle;

    m_max_outstanding_requests = 0;
    m_instCache_ptr = nullptr;
    m_dataCache_ptr = nullptr;

    m_instCache_ptr = p.icache;
    m_dataCache_ptr = p.dcache;
    m_max_outstanding_requests = p.max_outstanding_requests;
    m_deadlock_threshold = p.deadlock_threshold;

    assert(m_max_outstanding_requests > 0);
    assert(m_deadlock_threshold > 0);
    assert(m_instCache_ptr);
    assert(m_dataCache_ptr);

    m_runningGarnetStandalone = p.garnet_standalone;


    // These statistical variables are not for display.
    // The profiler will collate these across different
    // coalescers and display those collated statistics.
    m_outstandReqHist.init(10);
    m_latencyHist.init(10);
    m_missLatencyHist.init(10);

    for (int i = 0; i < RubyRequestType_NUM; i++) {
        m_typeLatencyHist.push_back(new statistics::Histogram());
        m_typeLatencyHist[i]->init(10);

        m_missTypeLatencyHist.push_back(new statistics::Histogram());
        m_missTypeLatencyHist[i]->init(10);
    }

    for (int i = 0; i < MachineType_NUM; i++) {
        m_missMachLatencyHist.push_back(new statistics::Histogram());
        m_missMachLatencyHist[i]->init(10);

        m_IssueToInitialDelayHist.push_back(new statistics::Histogram());
        m_IssueToInitialDelayHist[i]->init(10);

        m_InitialToForwardDelayHist.push_back(new statistics::Histogram());
        m_InitialToForwardDelayHist[i]->init(10);

        m_ForwardToFirstResponseDelayHist.push_back(
            new statistics::Histogram());
        m_ForwardToFirstResponseDelayHist[i]->init(10);

        m_FirstResponseToCompletionDelayHist.push_back(
            new statistics::Histogram());
        m_FirstResponseToCompletionDelayHist[i]->init(10);
    }

    for (int i = 0; i < RubyRequestType_NUM; i++) {
        m_missTypeMachLatencyHist.push_back(
            std::vector<statistics::Histogram *>());

        for (int j = 0; j < MachineType_NUM; j++) {
            m_missTypeMachLatencyHist[i].push_back(
                new statistics::Histogram());
            m_missTypeMachLatencyHist[i][j]->init(10);
        }
    }

}

GPUCoalescer::~GPUCoalescer()
{
}

Port &
GPUCoalescer::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "gmTokenPort") {
        return gmTokenPort;
    }

    // delgate to RubyPort otherwise
    return RubyPort::getPort(if_name, idx);
}

void
GPUCoalescer::wakeup()
{
    Cycles current_time = curCycle();
    for (auto& requestList : coalescedTable) {
        for (auto& req : requestList.second) {
            if (current_time - req->getIssueTime() > m_deadlock_threshold) {
                std::stringstream ss;
                printRequestTable(ss);
                warn("GPUCoalescer %d Possible deadlock detected!\n%s\n",
                     m_version, ss.str());
                panic("Aborting due to deadlock!\n");
            }
        }
    }

    Tick tick_threshold = cyclesToTicks(m_deadlock_threshold);
    uncoalescedTable.checkDeadlock(tick_threshold);

    if (m_outstanding_count > 0) {
        schedule(deadlockCheckEvent,
                 m_deadlock_threshold * clockPeriod() +
                 curTick());
    }
}

void
GPUCoalescer::printRequestTable(std::stringstream& ss)
{
    ss << "Printing out " << coalescedTable.size()
       << " outstanding requests in the coalesced table\n";

    for (auto& requestList : coalescedTable) {
        for (auto& request : requestList.second) {
            ss << "\tAddr: " << printAddress(requestList.first) << "\n"
               << "\tInstruction sequence number: "
               << request->getSeqNum() << "\n"
               << "\t\tType: "
               << RubyRequestType_to_string(request->getRubyType()) << "\n"
               << "\t\tNumber of associated packets: "
               << request->getPackets().size() << "\n"
               << "\t\tIssue time: "
               << request->getIssueTime() * clockPeriod() << "\n"
               << "\t\tDifference from current tick: "
               << (curCycle() - request->getIssueTime()) * clockPeriod()
               << "\n";
        }
    }

    // print out packets waiting to be issued in uncoalesced table
    uncoalescedTable.printRequestTable(ss);
}

/**
 * Emits the end-of-simulation FAULT_ANALYSIS width summary record for the L1
 * Vector Data Cache (TCP) when TCP width tracking is enabled.
 */
void
GPUCoalescer::exitCallback()
{
    if (!Shader::tcpWidthTrackingEnabledGlobal() ||
        tcpWidth.summaryPrinted) {
        return;
    }

    tcpWidth.summaryPrinted = true;

    inform(
        "FAULT_ANALYSIS schema=1 category=WIDTH component=TCP "
        "cache_id=%d set=%d way=%d byte=%d bit=%d stuck=%d "
        "total_accesses=%llu no_effect_accesses=%llu "
        "effective_activations=%llu accessed_unique_lane_contexts=%llu "
        "accessed_lane_index_union_count=%llu "
        "accessed_unique_wavefronts=%llu accessed_unique_simds=%llu "
        "accessed_unique_cus=%llu effective_unique_lane_contexts=%llu "
        "effective_lane_index_union_count=%llu "
        "effective_unique_wavefronts=%llu effective_unique_simds=%llu "
        "effective_unique_cus=%llu lane_semantics=REQUEST_LANE "
        "exposure_class=%s effective_exposure_class=%s",
        tcpWidth.faultCache,
        tcpWidth.faultSet,
        tcpWidth.faultWay,
        tcpWidth.faultByte,
        tcpWidth.faultBit,
        tcpWidth.stuckValue,
        static_cast<unsigned long long>(tcpWidth.totalAccesses),
        static_cast<unsigned long long>(tcpWidth.noEffectAccesses),
        static_cast<unsigned long long>(tcpWidth.effectiveActivations),
        static_cast<unsigned long long>(tcpWidth.accessedLaneContexts.size()),
        static_cast<unsigned long long>(tcpWidth.accessedLaneIndices.size()),
        static_cast<unsigned long long>(tcpWidth.accessedWavefronts.size()),
        static_cast<unsigned long long>(tcpWidth.accessedSimds.size()),
        static_cast<unsigned long long>(tcpWidth.accessedCus.size()),
        static_cast<unsigned long long>(tcpWidth.effectiveLaneContexts.size()),
        static_cast<unsigned long long>(tcpWidth.effectiveLaneIndices.size()),
        static_cast<unsigned long long>(tcpWidth.effectiveWavefronts.size()),
        static_cast<unsigned long long>(tcpWidth.effectiveSimds.size()),
        static_cast<unsigned long long>(tcpWidth.effectiveCus.size()),
        tcpExposureClass(
            tcpWidth.accessedCus,
            tcpWidth.accessedSimds,
            tcpWidth.accessedWavefronts,
            tcpWidth.accessedLaneContexts),
        tcpExposureClass(
            tcpWidth.effectiveCus,
            tcpWidth.effectiveSimds,
            tcpWidth.effectiveWavefronts,
            tcpWidth.effectiveLaneContexts));
}

void
GPUCoalescer::resetStats()
{
    m_latencyHist.reset();
    m_missLatencyHist.reset();
    for (int i = 0; i < RubyRequestType_NUM; i++) {
        m_typeLatencyHist[i]->reset();
        m_missTypeLatencyHist[i]->reset();
        for (int j = 0; j < MachineType_NUM; j++) {
            m_missTypeMachLatencyHist[i][j]->reset();
        }
    }

    for (int i = 0; i < MachineType_NUM; i++) {
        m_missMachLatencyHist[i]->reset();

        m_IssueToInitialDelayHist[i]->reset();
        m_InitialToForwardDelayHist[i]->reset();
        m_ForwardToFirstResponseDelayHist[i]->reset();
        m_FirstResponseToCompletionDelayHist[i]->reset();
    }
}

void
GPUCoalescer::printProgress(std::ostream& out) const
{
}

// sets the kernelEndList
void
GPUCoalescer::insertKernel(int wavefront_id, PacketPtr pkt)
{
    // Don't know if this will happen or is possible
    // but I just want to be careful and not have it become
    // simulator hang in the future
    DPRINTF(GPUCoalescer, "inserting wf: %d to kernelEndlist\n", wavefront_id);
    assert(kernelEndList.count(wavefront_id) == 0);

    kernelEndList[wavefront_id] = pkt;
    DPRINTF(GPUCoalescer, "kernelEndList->size() = %d\n",
            kernelEndList.size());
}

void
GPUCoalescer::writeCallback(Addr address, DataBlock& data)
{
    writeCallback(address, MachineType_NULL, data);
}

void
GPUCoalescer::writeCallback(Addr address,
                         MachineType mach,
                         DataBlock& data)
{
    writeCallback(address, mach, data, Cycles(0), Cycles(0), Cycles(0));
}

void
GPUCoalescer::writeCallback(Addr address,
                         MachineType mach,
                         DataBlock& data,
                         Cycles initialRequestTime,
                         Cycles forwardRequestTime,
                         Cycles firstResponseTime)
{
    writeCallback(address, mach, data,
                  initialRequestTime, forwardRequestTime, firstResponseTime,
                  false);
}

void
GPUCoalescer::writeCallback(Addr address,
                         MachineType mach,
                         DataBlock& data,
                         Cycles initialRequestTime,
                         Cycles forwardRequestTime,
                         Cycles firstResponseTime,
                         bool isRegion)
{
    assert(address == makeLineAddress(address));
    assert(coalescedTable.count(address));

    auto crequest = coalescedTable.at(address).front();

    hitCallback(crequest, mach, data, true, crequest->getIssueTime(),
                forwardRequestTime, firstResponseTime, isRegion, false, false);

    // remove this crequest in coalescedTable
    delete crequest;
    coalescedTable.at(address).pop_front();

    if (coalescedTable.at(address).empty()) {
        coalescedTable.erase(address);
    } else {
        auto nextRequest = coalescedTable.at(address).front();
        issueRequest(nextRequest);
    }
}

void
GPUCoalescer::writeCompleteCallback(Addr address,
                                    uint64_t instSeqNum,
                                    MachineType mach)
{
    DPRINTF(GPUCoalescer, "writeCompleteCallback for address 0x%x"
            " instSeqNum = %d\n", address, instSeqNum);

    assert(pendingWriteInsts.count(instSeqNum) == 1);
    PendingWriteInst& inst = pendingWriteInsts[instSeqNum];

    // check the uncoalescedTable to see whether all requests for the inst
    // have been issued or not
    bool reqsAllIssued = uncoalescedTable.areRequestsDone(instSeqNum);
    DPRINTF(GPUCoalescer, "instSeqNum = %d, pendingStores=%d, "
                    "reqsAllIssued=%d\n", reqsAllIssued,
                    inst.getNumPendingStores()-1, reqsAllIssued);

    if (inst.receiveWriteCompleteAck() && reqsAllIssued ) {
        // if the pending write instruction has received all write completion
        // callbacks for its issued Ruby requests, we can now start respond
        // the requesting CU in one response packet.
        inst.ackWriteCompletion(m_usingRubyTester);

        DPRINTF(GPUCoalescer, "write inst %d completed at coalescer\n",
                instSeqNum);
        pendingWriteInsts.erase(instSeqNum);
    }
}

void
GPUCoalescer::readCallback(Addr address, DataBlock& data)
{
    readCallback(address, MachineType_NULL, data, false);
}

void
GPUCoalescer::readCallback(Addr address,
                        MachineType mach,
                        DataBlock& data,
                        bool externalHit = false)
{
    readCallback(address, mach, data, Cycles(0), Cycles(0), Cycles(0), externalHit);
}

void
GPUCoalescer::readCallback(Addr address,
                        MachineType mach,
                        DataBlock& data,
                        Cycles initialRequestTime,
                        Cycles forwardRequestTime,
                        Cycles firstResponseTime,
                        bool externalHit = false)
{

    readCallback(address, mach, data,
                 initialRequestTime, forwardRequestTime, firstResponseTime,
                 false, externalHit);
}

void
GPUCoalescer::readCallback(Addr address,
                        MachineType mach,
                        DataBlock& data,
                        Cycles initialRequestTime,
                        Cycles forwardRequestTime,
                        Cycles firstResponseTime,
                        bool isRegion,
                        bool externalHit)
{
// ---- FAULT DETECTION (TCP + TCC) ----
bool block_poisoned = false;

bool tcp_hard_faulted = false;
bool tcp_transient_faulted = false;

// Populated by CacheMemory when the returned line maps to the faulted
// physical TCP slot.
CacheMemory::TcpHardFaultHit tcp_hard_hit;

// Check the local L1/TCP for permanent faults.
//
// Permanent faults must also be checked when a load completes after
// a TCP miss/fill. enforceFaultOnLoad() verifies that the returned
// address is actually resident in the physical TCP set/way, so bypass
// responses and evicted lines remain unaffected.
if (m_dataCache_ptr) {
    tcp_hard_faulted =
        m_dataCache_ptr->enforceFaultOnLoad(
            address,
            data,
            &tcp_hard_hit);

    // Preserve the existing transient-fault behavior for now.
    if (externalHit) {
        tcp_transient_faulted =
            m_dataCache_ptr->transientFaultAffectsLoad(address);
    }

    block_poisoned =
        tcp_hard_faulted || tcp_transient_faulted;
}

    Addr line_addr = makeLineAddress(address);
bool tcc_poisoned = CacheMemory::isTccPoisoned(line_addr);


if (tcc_poisoned) {
    block_poisoned = true;
    inform("[TCC-POISON-CONSUME-GPUCOAL] line=%#lx addr=%#lx Tagging payload.",
           line_addr, address);
}

    assert(address == makeLineAddress(address));
    assert(coalescedTable.count(address));

    auto crequest = coalescedTable.at(address).front();
    fatal_if(crequest->getRubyType() != RubyRequestType_LD,
             "readCallback received non-read type response\n");

    bool mshr_hit_under_miss = false;

    while (crequest->getRubyType() == RubyRequestType_LD) {
      // ------------------------------------------------------------
// TCP first-consumer tracing.
//
// Does not modify the packet or payload data here; instead, it
// identifies which original GPU request packets overlap the exact
// faulted byte within the returned 64-byte cache line.
// ------------------------------------------------------------
if (Shader::tcpWidthTrackingEnabledGlobal() &&
    tcp_hard_faulted &&
    tcp_hard_hit.matched &&
    !m_usingRubyTester) {

    for (auto& pkt : crequest->getPackets()) {
        const uint32_t packet_offset =
            getOffset(pkt->getAddr());

        const uint32_t packet_size =
            pkt->getSize();

        const uint32_t packet_end =
            packet_offset + packet_size;

        // Example:
        // fault byte = 26
        // packet bytes = 24..27
        // Therefore, this specific execution lane consumed the faulty byte.
        const bool overlaps_fault_byte =
            tcp_hard_hit.byte >=
                static_cast<int>(packet_offset) &&
            tcp_hard_hit.byte <
                static_cast<int>(packet_end);

        if (!overlaps_fault_byte) {
            continue;
        }

        RubyPort::SenderState *ruby_state =
            safe_cast<RubyPort::SenderState*>(
                pkt->senderState);

        ComputeUnit::DataPort::SenderState *cu_state =
            safe_cast<ComputeUnit::DataPort::SenderState*>(
                ruby_state->predecessor);

        GPUDynInstPtr inst =
            cu_state->_gpuDynInst;

        if (!inst) {
            continue;
        }

        const int lane =
            static_cast<int>(cu_state->port_index);

        if (Shader::propagationVerboseEnabledGlobal()) {
        inform(
            "FAULT_TRACE event=TCP_CONSUMER fault_id=0 "
            "tick=%llu cache_id=%d "
            "line=0x%llx packet_addr=0x%llx "
            "packet_size=%u "
            "fault_set=%d fault_way=%d "
            "fault_byte=%d fault_bit=%d stuck=%d "
            "byte_in_packet=%d "
            "destination_cu=%d "
            "destination_simd=%d "
            "destination_wf_slot=%d "
            "destination_wf_dyn=%d "
            "lane=%d "
            "inst_seq=%llu "
            "pc=0x%llx "
            "instruction=\"%s\"",
            static_cast<unsigned long long>(curTick()),
            m_version,
            static_cast<unsigned long long>(line_addr),
            static_cast<unsigned long long>(
                pkt->getAddr()),
            static_cast<unsigned int>(packet_size),
            tcp_hard_hit.set,
            tcp_hard_hit.way,
            tcp_hard_hit.byte,
            tcp_hard_hit.bit,
            tcp_hard_hit.stuck_val,
            tcp_hard_hit.byte -
                static_cast<int>(packet_offset),
            inst->cu_id,
            inst->simdId,
            inst->wfSlotId,
            inst->wfDynId,
            lane,
            static_cast<unsigned long long>(
                inst->seqNum()),
            static_cast<unsigned long long>(
                inst->pc()),
            inst->disassemble().c_str());
        }

        recordTcpWidth(
            m_version,
            tcp_hard_hit,
            inst->cu_id,
            inst->simdId,
            inst->wfSlotId,
            inst->wfDynId,
            lane);
    }
}



        // ---- MARK THE PACKETS IF POISONED ----
        if (block_poisoned) {
            for (auto& pkt : crequest->getPackets()) {
                pkt->setHasPoisonedPayload();
            }
        } else {
            for (auto& pkt : crequest->getPackets()) {
                pkt->clearHasPoisonedPayload();
            }
        }
        // -----------------------------------------
        hitCallback(crequest, mach, data, true,
                crequest->getIssueTime(), forwardRequestTime, firstResponseTime,
                isRegion, externalHit, mshr_hit_under_miss);

        delete crequest;
        coalescedTable.at(address).pop_front();
        if (coalescedTable.at(address).empty()) {
            break;
        }

        crequest = coalescedTable.at(address).front();

        PacketPtr pkt = crequest->getFirstPkt();
        bool is_request_local = !pkt->isGLCSet() && !pkt->isSLCSet();
        if (!is_request_local) {
            break;
        }

        mshr_hit_under_miss = true;
    }

    if (coalescedTable.at(address).empty()) {
        coalescedTable.erase(address);
    } else {
        auto nextRequest = coalescedTable.at(address).front();
        issueRequest(nextRequest);
    }
}

void
GPUCoalescer::hitCallback(CoalescedRequest* crequest,
                       MachineType mach,
                       DataBlock& data,
                       bool success,
                       Cycles initialRequestTime,
                       Cycles forwardRequestTime,
                       Cycles firstResponseTime,
                       bool isRegion,
                       bool externalHit = false,
                       bool mshrHitUnderMiss = false)
{
    PacketPtr pkt = crequest->getFirstPkt();
    Addr request_address = pkt->getAddr();
    [[maybe_unused]] Addr request_line_address =
        makeLineAddress(request_address);

    RubyRequestType type = crequest->getRubyType();

    DPRINTF(GPUCoalescer, "Got hitCallback for 0x%X\n", request_line_address);

    DPRINTF(RubyHitMiss, "GPU TCP Cache %s at %#x\n",
                        externalHit ? "hit" : "miss",
                        printAddress(request_address));

    recordStats(crequest, mach,
                      initialRequestTime,
                      forwardRequestTime,
                      firstResponseTime,
                      isRegion,
                      mshrHitUnderMiss);
    // update the data
    //
    // MUST ADD DOING THIS FOR EACH REQUEST IN COALESCER
    std::vector<PacketPtr> pktList = crequest->getPackets();

    uint8_t* log = nullptr;
    DPRINTF(GPUCoalescer, "Responding to %d packets for addr 0x%X\n",
            pktList.size(), request_line_address);
    uint32_t offset;
    int pkt_size;
    for (auto& pkt : pktList) {
        offset = getOffset(pkt->getAddr());
        pkt_size = pkt->getSize();
        request_address = pkt->getAddr();

        // When the Ruby system is cooldown phase, the requests come from
        // the cache recorder. These requests do not get coalesced and
        // do not return valid data.
        if (m_ruby_system->getCooldownEnabled())
            continue;

        if (pkt->getPtr<uint8_t>()) {
            switch(type) {
                // Store and AtomicNoReturns follow the same path, as the
                // data response is not needed.
                case RubyRequestType_ATOMIC_NO_RETURN:
                    assert(pkt->isAtomicOp());
                    break;
                case RubyRequestType_ST:
                    break;
              case RubyRequestType_LD:
    pkt->setData(data.getData(offset, pkt_size));



    break;
                case RubyRequestType_ATOMIC_RETURN:
                    assert(pkt->isAtomicOp());
                    // Atomic operations are performed by the WriteMask
                    // in packet order, set by the crequest. Thus, when
                    // unpacking the changes from the log, we read from
                    // the front of the log to correctly map response
                    // data into the packets.

                    // Log entry contains the old value before the current
                    // atomic operation occurred.
                    log = data.popAtomicLogEntryFront();
                    pkt->setData(&log[offset]);
                    delete [] log;
                    log = nullptr;
                    break;
                default:
                    panic("Unsupported ruby packet type:%s\n",
                                    RubyRequestType_to_string(type));
                    break;
            }
        } else {
            DPRINTF(MemoryAccess,
                    "WARNING.  Data not transfered from Ruby to M5 for type " \
                    "%s\n",
                    RubyRequestType_to_string(type));
        }
    }
    assert(data.numAtomicLogEntries() == 0);

    m_outstanding_count--;
    assert(m_outstanding_count >= 0);

    completeHitCallback(pktList);
}

bool
GPUCoalescer::empty() const
{
    return coalescedTable.empty();
}

RubyRequestType
GPUCoalescer::getRequestType(PacketPtr pkt)
{
    RubyRequestType req_type = RubyRequestType_NULL;

    // These types are not support or not used in GPU caches.
    assert(!pkt->req->isLLSC());
    assert(!pkt->req->isLockedRMW());
    assert(!pkt->req->isInstFetch());

    if (pkt->req->isAtomicReturn()) {
        req_type = RubyRequestType_ATOMIC_RETURN;
    } else if (pkt->req->isAtomicNoReturn()) {
        req_type = RubyRequestType_ATOMIC_NO_RETURN;
    } else if (pkt->isRead()) {
        req_type = RubyRequestType_LD;
    } else if (pkt->isWrite()) {
        req_type = RubyRequestType_ST;
    } else if (pkt->isFlush()) {
        req_type = RubyRequestType_FLUSH;
    } else {
        panic("Unsupported ruby packet type\n");
    }

    return req_type;
}

// Places an uncoalesced packet in uncoalescedTable. If the packet is a
// special type (MemFence, scoping, etc), it is issued immediately.
RequestStatus
GPUCoalescer::makeRequest(PacketPtr pkt)
{
    if (pkt->cmd == MemCmd::MemSyncReq) {
        // issue mem_sync requests immediately to the cache system without
        // going through uncoalescedTable like normal LD/ST/Atomic requests
        issueMemSyncRequest(pkt);
    } else {
        // all packets must have valid instruction sequence numbers
        assert(pkt->req->hasInstSeqNum());

        // otherwise, this must be either read or write command
        assert(pkt->isRead() || pkt->isWrite() || pkt->isFlush());

        InstSeqNum seq_num = pkt->req->getReqInstSeqNum();

        // in the case of protocol tester, there is one packet per sequence
        // number. The number of packets during simulation depends on the
        // number of lanes actives for that vmem request (i.e., the popcnt
        // of the exec_mask.
        int num_packets = 1;

        // When Ruby is in warmup or cooldown phase, the requests come from
        // the cache recorder. There is no dynamic instruction associated
        // with these requests either
        if (!m_ruby_system->getWarmupEnabled()
                && !m_ruby_system->getCooldownEnabled()) {
            if (!m_usingRubyTester) {
                num_packets = 0;
                for (int i = 0; i < TheGpuISA::NumVecElemPerVecReg; i++) {
                    num_packets += getDynInst(pkt)->getLaneStatus(i);
                }
            }
        }

        // the pkt is temporarily stored in the uncoalesced table until
        // it's picked for coalescing process later in this cycle or in a
        // future cycle. Packets remaining is set to the number of excepted
        // requests from the instruction based on its exec_mask.
        uncoalescedTable.insertPacket(pkt);
        uncoalescedTable.insertReqType(pkt, getRequestType(pkt));
        uncoalescedTable.initPacketsRemaining(seq_num, num_packets);
        DPRINTF(GPUCoalescer, "Put pkt with addr 0x%X to uncoalescedTable\n",
                pkt->getAddr());

        // we schedule an issue event here to process the uncoalesced table
        // and try to issue Ruby request to cache system
        if (!issueEvent.scheduled()) {
            DPRINTF(GPUCoalescer, "Scheduled issueEvent for seqNum %d\n",
                    seq_num);
            schedule(issueEvent, curTick());
        }
    }

    // we always return RequestStatus_Issued in this coalescer
    // b/c the coalescer's resouce was checked ealier and the coalescer is
    // queueing up aliased requets in its coalesced table
    return RequestStatus_Issued;
}

template <class KEY, class VALUE>
std::ostream &
operator<<(std::ostream &out, const std::unordered_map<KEY, VALUE> &map)
{
    out << "[";
    for (auto i = map.begin(); i != map.end(); ++i)
        out << " " << i->first << "=" << i->second;
    out << " ]";

    return out;
}

void
GPUCoalescer::print(std::ostream& out) const
{
    out << "[GPUCoalescer: " << m_version
        << ", outstanding requests: " << m_outstanding_count
        << "]";
}

GPUDynInstPtr
GPUCoalescer::getDynInst(PacketPtr pkt) const
{
    RubyPort::SenderState* ss =
            safe_cast<RubyPort::SenderState*>(pkt->senderState);

    ComputeUnit::DataPort::SenderState* cu_state =
        safe_cast<ComputeUnit::DataPort::SenderState*>
            (ss->predecessor);

    return cu_state->_gpuDynInst;
}

bool
GPUCoalescer::coalescePacket(PacketPtr pkt)
{
    uint64_t seqNum = pkt->req->getReqInstSeqNum();
    Addr line_addr = makeLineAddress(pkt->getAddr());

    // If the packet has the same line address as a request already in the
    // coalescedTable and has the same sequence number, it can be coalesced.
    if (coalescedTable.count(line_addr)) {
        // Search for a previous coalesced request with the same seqNum.
        auto& creqQueue = coalescedTable.at(line_addr);
        auto citer = std::find_if(creqQueue.begin(), creqQueue.end(),
            [&](CoalescedRequest* c) { return c->getSeqNum() == seqNum; }
        );
        if (citer != creqQueue.end()) {
            (*citer)->insertPacket(pkt);
            return true;
        }
    }

    if (m_outstanding_count < m_max_outstanding_requests) {
        // This is an "aliased" or new request. Create a RubyRequest and
        // append it to the list of "targets" in the coalescing table.
        DPRINTF(GPUCoalescer, "Creating new or aliased request for 0x%X\n",
                line_addr);

        CoalescedRequest *creq = new CoalescedRequest(seqNum);
        creq->insertPacket(pkt);
        creq->setRubyType(getRequestType(pkt));
        creq->setIssueTime(curCycle());

        if (!coalescedTable.count(line_addr)) {
            // If there is no outstanding request for this line address,
            // create a new coalecsed request and issue it immediately.
            auto reqList = std::deque<CoalescedRequest*> { creq };
            coalescedTable.insert(std::make_pair(line_addr, reqList));
            if (!coalescedReqs.count(seqNum)) {
                coalescedReqs.insert(std::make_pair(seqNum, reqList));
            } else {
                coalescedReqs.at(seqNum).push_back(creq);
            }
        } else {
            // The request is for a line address that is already outstanding
            // but for a different instruction. Add it as a new request to be
            // issued when the current outstanding request is completed.
            coalescedTable.at(line_addr).push_back(creq);
            DPRINTF(GPUCoalescer, "found address 0x%X with new seqNum %d\n",
                    line_addr, seqNum);
        }

        // In both cases, requests are added to the coalescing table and will
        // be counted as outstanding requests.
        m_outstanding_count++;

        // We track all issued or to-be-issued Ruby requests associated with
        // write instructions. An instruction may have multiple Ruby
        // requests.
        if (pkt->cmd == MemCmd::WriteReq) {
            DPRINTF(GPUCoalescer, "adding write inst %d at line 0x%x to"
                    " the pending write instruction list\n", seqNum,
                    line_addr);

            RubyPort::SenderState* ss =
                    safe_cast<RubyPort::SenderState*>(pkt->senderState);

            // we need to save this port because it will be used to call
            // back the requesting CU when we receive write
            // complete callbacks for all issued Ruby requests of this
            // instruction.
            RubyPort::MemResponsePort* mem_response_port = ss->port;

            GPUDynInstPtr gpuDynInst = nullptr;

            if (!m_usingRubyTester) {
                // If this coalescer is connected to a real CU, we need
                // to save the corresponding gpu dynamic instruction.
                // CU will use that instruction to decrement wait counters
                // in the issuing wavefront.
                // For Ruby tester, gpuDynInst == nullptr
                gpuDynInst = getDynInst(pkt);
            }

            PendingWriteInst& inst = pendingWriteInsts[seqNum];
            inst.addPendingReq(mem_response_port, gpuDynInst,
                               m_usingRubyTester);
        }

        return true;
    }

    // The maximum number of outstanding requests have been issued.
    return false;
}

void
GPUCoalescer::completeIssue()
{
    // Iterate over the maximum number of instructions we can coalesce
    // per cycle (coalescingWindow).
    for (int instIdx = 0; instIdx < coalescingWindow; ++instIdx) {
        PerInstPackets *pkt_list =
            uncoalescedTable.getInstPackets(instIdx);

        // getInstPackets will return nullptr if no instruction
        // exists at the current offset.
        if (!pkt_list) {
            break;
        } else if (pkt_list->empty()) {
            // Found something, but it has not been cleaned up by update
            // resources yet. See if there is anything else to coalesce.
            // Assume we can't check anymore if the coalescing window is 1.
            continue;
        } else {
            // All packets in the list have the same seqNum, use first.
            InstSeqNum seq_num = pkt_list->front()->req->getReqInstSeqNum();

            // The difference in list size before and after tells us the
            // number of packets which were coalesced.
            size_t pkt_list_size = pkt_list->size();

            // Since we have a pointer to the list of packets in the inst,
            // erase them from the list if coalescing is successful and
            // leave them in the list otherwise. This aggressively attempts
            // to coalesce as many packets as possible from the current inst.
            pkt_list->remove_if(
                [&](PacketPtr pkt) { return coalescePacket(pkt); }
            );

            if (coalescedReqs.count(seq_num)) {
                auto& creqs = coalescedReqs.at(seq_num);
                for (auto creq : creqs) {
                    DPRINTF(GPUCoalescer, "Issued req type %s seqNum %d\n",
                            RubyRequestType_to_string(creq->getRubyType()),
                                                      seq_num);
                    issueRequest(creq);
                }
                coalescedReqs.erase(seq_num);
            }

            assert(pkt_list_size >= pkt_list->size());
            size_t pkt_list_diff = pkt_list_size - pkt_list->size();

            int num_remaining = uncoalescedTable.getPacketsRemaining(seq_num);
            num_remaining -= pkt_list_diff;
            assert(num_remaining >= 0);

            uncoalescedTable.setPacketsRemaining(seq_num, num_remaining);
            DPRINTF(GPUCoalescer,
                    "Coalesced %d pkts for seqNum %d, %d remaining\n",
                    pkt_list_diff, seq_num, num_remaining);
        }
    }

    // Clean up any instructions in the uncoalesced table that have had
    // all of their packets coalesced and return a token for that column.
    uncoalescedTable.updateResources();

    // have Kernel End releases been issued this cycle
    int len = newKernelEnds.size();
    for (int i = 0; i < len; i++) {
        kernelCallback(newKernelEnds[i]);
    }
    newKernelEnds.clear();
}

void
GPUCoalescer::evictionCallback(Addr address)
{
    ruby_eviction_callback(address);
}

void
GPUCoalescer::kernelCallback(int wavefront_id)
{
    assert(kernelEndList.count(wavefront_id));

    ruby_hit_callback(kernelEndList[wavefront_id]);

    kernelEndList.erase(wavefront_id);
}

void
GPUCoalescer::atomicCallback(Addr address,
                             MachineType mach,
                             const DataBlock& data)
{
    assert(address == makeLineAddress(address));
    assert(coalescedTable.count(address));

    auto crequest = coalescedTable.at(address).front();

    fatal_if((crequest->getRubyType() != RubyRequestType_ATOMIC &&
              crequest->getRubyType() != RubyRequestType_ATOMIC_RETURN &&
              crequest->getRubyType() != RubyRequestType_ATOMIC_NO_RETURN),
             "atomicCallback saw non-atomic type response\n");

    hitCallback(crequest, mach, (DataBlock&)data, true,
                crequest->getIssueTime(), Cycles(0), Cycles(0), false, false);

    delete crequest;
    coalescedTable.at(address).pop_front();

    if (coalescedTable.at(address).empty()) {
        coalescedTable.erase(address);
    } else {
        auto nextRequest = coalescedTable.at(address).front();
        issueRequest(nextRequest);
    }
}

void
GPUCoalescer::completeHitCallback(std::vector<PacketPtr> & mylist)
{
    for (auto& pkt : mylist) {
        // When Ruby is in warmup or cooldown phase, the requests come
        // from the cache recorder. They do not track which port to use
        // and do not need to send the response back
        if (!m_ruby_system->getWarmupEnabled()
                && !m_ruby_system->getCooldownEnabled()) {
            RubyPort::SenderState *ss =
                safe_cast<RubyPort::SenderState *>(pkt->senderState);
            MemResponsePort *port = ss->port;
            assert(port != NULL);

            pkt->senderState = ss->predecessor;

            if (pkt->cmd != MemCmd::WriteReq) {
                // for WriteReq, we keep the original senderState until
                // writeCompleteCallback
                delete ss;
            }

            port->hitCallback(pkt);
            trySendRetries();
        }
    }

    // We schedule an event in the same tick as hitCallback (similar to
    // makeRequest) rather than calling completeIssue directly to reduce
    // function calls to complete issue. This can only happen if the max
    // outstanding requests is less than the number of slots in the
    // uncoalesced table and makeRequest is not called again.
    if (uncoalescedTable.packetAvailable() && !issueEvent.scheduled()) {
        schedule(issueEvent, curTick());
    }

    RubySystem *rs = m_ruby_system;
    if (m_ruby_system->getWarmupEnabled()) {
        rs->m_cache_recorder->enqueueNextFetchRequest();
    } else if (m_ruby_system->getCooldownEnabled()) {
        rs->m_cache_recorder->enqueueNextFlushRequest();
    } else {
        testDrainComplete();
    }
}

void
GPUCoalescer::recordStats(CoalescedRequest* crequest,
                                MachineType mach,
                                Cycles initialRequestTime,
                                Cycles forwardRequestTime,
                                Cycles firstResponseTime,
                                bool isRegion, bool mshrHitUnderMiss)
{
    RubyRequestType type = crequest->getRubyType();

    if (mshrHitUnderMiss) {
        // Add the number of mshr hits under misses to the
        // TCP demand hits stat.
        // We don't need to profile misses since they will be
        // profiled at the TCP. Only the MSHR hits under misses
        // needs to be profiled here
        PacketPtr pkt = crequest->getFirstPkt();
        if (!pkt->isGLCSet() &&
                !pkt->isSLCSet()) {
            m_dataCache_ptr->profileDemandHit();
        }

        // Since the request hit in the mshr, update mshr stats
        if (type == RubyRequestType_LD) {
            stats.m_mshr_ld_hits_under_miss++;
        }
    } else  {
        if (type == RubyRequestType_LD) {
            stats.m_mshr_ld_misses++;
        } else {
            stats.m_mshr_st_misses++;
        }
    }
}

GPUCoalescer::GPUCoalescerStats::GPUCoalescerStats(statistics::Group *parent)
    : statistics::Group(parent),
    ADD_STAT(m_mshr_ld_hits_under_miss,
            "Number of load requests that hit in the coalescer MSHR"),
    ADD_STAT(m_mshr_ld_misses,
            "Number of load requests that miss in the coalescer MSHR"),
    ADD_STAT(m_mshr_st_misses,
            "Number of store requests that miss in the coalescer MSHR"),
    ADD_STAT(m_mshr_accesses,
            "Number of mshr accesses",
            m_mshr_ld_hits_under_miss + m_mshr_ld_misses
            + m_mshr_st_misses)
{
}

/** Directly injects a single-bit flip into the attached TCP CacheMemory array. */
void
GPUCoalescer::injectValue(int set, int way, int byte, int bit)
{
    if (m_dataCache_ptr) {
        // Pass the coordinates down to the physical cache memory array
        m_dataCache_ptr->injectBitFlip(set, way, byte, bit);
    } else {
        panic("GPUCoalescer: No TCP data cache attached for fault injection!");
    }
}

/** Forwards a permanent stuck-at (SA0/SA1) fault descriptor to the TCP CacheMemory. */
void GPUCoalescer::loadTcpHardFault(int set, int way, int byte, int bit, int stuck_val) {
    inform("GPUCoalescer: Received HARD TCP fault. DataCache attached? %s", m_dataCache_ptr ? "YES" : "NO");

    if (m_dataCache_ptr) {
        m_dataCache_ptr->loadTcpHardFault(set, way, byte, bit, stuck_val);
    }
}

/** Triggers the timed activation of a preloaded transient TCP fault if the target line is valid. */
void
GPUCoalescer::activateTcpTransientFault(int set, int way)
{
    if (m_dataCache_ptr) {
        m_dataCache_ptr->activateTcpTransientFaultIfMatch(set, way);
    } else {
        warn("GPUCoalescer: Cannot activate transient TCP fault, no data cache attached.");
    }
}

/** Preloads transient TCP fault coordinates into the TCP CacheMemory prior to timed activation. */
void
GPUCoalescer::loadTcpTransientFault(int set, int way, int byte, int bit)
{
    inform("GPUCoalescer: Received TRANSIENT TCP fault. DataCache attached? %s",
           m_dataCache_ptr ? "YES" : "NO");

    if (m_dataCache_ptr) {
        m_dataCache_ptr->loadTcpTransientFault(set, way, byte, bit);
    }
}



} // namespace ruby
} // namespace gem5
