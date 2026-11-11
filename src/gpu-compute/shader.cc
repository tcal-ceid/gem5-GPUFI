/*
 * Copyright (c) 2011-2015 Advanced Micro Devices, Inc.
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

#include "gpu-compute/shader.hh"
#include "gpu-compute/scalar_register_file.hh"

#include <limits>

#include "gpu-compute/vector_register_file.hh"
#include "gpu-compute/shader.hh"
#include "arch/amdgpu/common/gpu_translation_state.hh"
#include "arch/amdgpu/common/tlb.hh"
#include "base/chunk_generator.hh"
#include "debug/GPUAgentDisp.hh"
#include "debug/GPUDisp.hh"
#include "debug/GPUMem.hh"
#include "debug/GPUShader.hh"
#include "debug/GPUWgLatency.hh"
#include "dev/amdgpu/hwreg_defines.hh"
#include "gpu-compute/dispatcher.hh"
#include "gpu-compute/gpu_command_processor.hh"
#include "gpu-compute/gpu_static_inst.hh"
#include "gpu-compute/hsa_queue_entry.hh"
#include "gpu-compute/wavefront.hh"
#include "mem/packet.hh"
#include "mem/ruby/system/RubySystem.hh"
#include "sim/sim_exit.hh"

  namespace gem5
  {

bool Shader::globalTcpFaultWidthTracking = false;
bool Shader::globalSqcFaultWidthTracking = false;
bool Shader::globalFaultPropagationVerbose = false;

Shader::Shader(const Params &p) : ClockedObject(p),
    _gpuVmApe({0, 0}),
    _ldsApe({0, 0}),
    _scratchApe({0, 0}),
    shHiddenPrivateBaseVmid(0),
    faultMask(p.fault_mask),
    faultMaskId(p.fault_mask_id),
    enableVrfFaultWidthTracking(
        p.enable_vrf_fault_width_tracking),
    enableVrfFaultDepthTracking(
        p.enable_vrf_fault_depth_tracking),
    enableSrfFaultWidthTracking(
        p.enable_srf_fault_width_tracking),
    enableSrfFaultDepthTracking(
        p.enable_srf_fault_depth_tracking),
    enableTcpFaultWidthTracking(
        p.enable_tcp_fault_width_tracking),
    enableSqcFaultWidthTracking(
        p.enable_sqc_fault_width_tracking),
    enableFaultPropagationVerbose(
        p.enable_fault_propagation_verbose),
    nextRegisterPropagationEventId(1),
    vrfFaults(),
    srfFaults(),
    tcpFaults(),
    tcpHardFaults(),
    faultsScheduled(false),
    _activeCus(0),
    _lastInactiveTick(0),
    kernelExitRequested(false),
    blitKernel(false),
    num_outstanding_invl2s(0),
    cpuThread(nullptr),
    gpuTc(nullptr),
    cpuPointer(p.cpu_pointer),
    tickEvent([this]{ execScheduledAdds(); }, "Shader scheduled adds event",
          false, Event::CPU_Tick_Pri),
    timingSim(p.timing),
    hsail_mode(SIMT),
    impl_kern_launch_acq(p.impl_kern_launch_acq),
    impl_kern_end_rel(p.impl_kern_end_rel),
    coissue_return(1),
    trace_vgpr_all(1),
    n_cu((p.CUs).size()),
    n_wf(p.n_wf),
    n_cu_per_sqc(p.cu_per_sqc),
    globalMemSize(p.globalmem),
    nextSchedCu(0),
    sa_n(0),
    tccFaults(),
    tccHardFaults(),
    sa_val(0),
    sa_when(0),
    sa_x(0),
    cuList(n_cu),
    gpuCmdProc(*p.gpu_cmd_proc),
    _dispatcher(*p.dispatcher),
    systemHub(p.system_hub),
    max_valu_insts(p.max_valu_insts),
    total_valu_insts(0),
    sqcFaults(),
    sqcHardFaults(),
    progressInterval(p.progress_interval),
    stats(this, p.CUs[0]->wfSize())
{
    globalTcpFaultWidthTracking = enableTcpFaultWidthTracking;
    globalSqcFaultWidthTracking = enableSqcFaultWidthTracking;
    globalFaultPropagationVerbose = enableFaultPropagationVerbose;

    for (auto *c : p.tcc_caches) {
        tccCaches.push_back(c);
    }

    if (!faultMask.empty()) {
        inform("Fault injection enabled: mask=%s id=%d\n",
               faultMask, faultMaskId);
    }




    // This processes the vrf_faults we packed in apu_se.py
    // loadVrfFaultMask();
   //  scheduleVrfFaults();


    gpuCmdProc.setShader(this);
    _dispatcher.setShader(this);

    // These apertures are set by the driver. In full system mode that is done
    // using a PM4 packet but the emulated SE mode driver does not set them
    // explicitly, so we need to define some reasonable defaults here.
    _gpuVmApe.base = ((Addr)1 << 61) + 0x1000000000000L;
    _gpuVmApe.limit = (_gpuVmApe.base & 0xFFFFFF0000000000UL) | 0xFFFFFFFFFFL;

    _ldsApe.base = 0x1000000000000;
    _ldsApe.limit =  (_ldsApe.base & 0xFFFFFFFF00000000UL) | 0xFFFFFFFF;

    _scratchApe.base = 0x2000000000000;
    _scratchApe.limit = (_scratchApe.base & 0xFFFFFFFF00000000UL) | 0xFFFFFFFF;

    // The scratch and LDS address can be queried starting in gfx900. The
    // base addresses are in the SH_MEM_BASES 32-bit register. The upper 16
    // bits are for the LDS address and the lower 16 bits are for scratch
    // address. In both cases the 16 bits represent bits 63:48 of the address.
    // This means bits 47:0 of the base address is always zero.
    setHwReg(HW_REG_SH_MEM_BASES, 0x00010002);

    shHiddenPrivateBaseVmid = 0;

    cuList.resize(n_cu);

    panic_if(n_wf <= 0, "Must have at least 1 WF Slot per SIMD");

    for (int i = 0; i < n_cu; ++i) {
        cuList[i] = p.CUs[i];
        assert(i == cuList[i]->cu_id);
        cuList[i]->shader = this;
        cuList[i]->idleCUTimeout = p.idlecu_timeout;
    }
}

GPUDispatcher&
Shader::dispatcher()
{
    return _dispatcher;
}

Addr
Shader::mmap(int length)
{

    Addr start;

    // round up length to the next page
    length = roundUp(length, X86ISA::PageBytes);

    Process *proc = gpuTc->getProcessPtr();
    auto mem_state = proc->memState;

    if (proc->mmapGrowsDown()) {
        DPRINTF(GPUShader, "GROWS DOWN");
        start = mem_state->getMmapEnd() - length;
        mem_state->setMmapEnd(start);
    } else {
        DPRINTF(GPUShader, "GROWS UP");
        start = mem_state->getMmapEnd();
        mem_state->setMmapEnd(start + length);

        // assertion to make sure we don't overwrite the stack (it grows down)
        assert(mem_state->getStackBase() - mem_state->getMaxStackSize() >
               mem_state->getMmapEnd());
    }

    DPRINTF(GPUShader, "Shader::mmap start= %#x, %#x\n", start, length);

    proc->allocateMem(start, length);

    return start;
}

void
Shader::init()
{
    // grab the threadContext of the thread running on the CPU
    assert(cpuPointer);
    gpuTc = cpuPointer->getContext(0);
    assert(gpuTc);
}

uint64_t
Shader::allocateRegisterPropagationEventId()
{
    if (!enableVrfFaultDepthTracking && !enableSrfFaultDepthTracking) {
        return 0;
    }

    const uint64_t eventId = nextRegisterPropagationEventId++;

    // Zero is permanently reserved for clean/no provenance.
    assert(eventId != 0);

    return eventId;
}

Shader::~Shader()
{
    for (int j = 0; j < n_cu; ++j)
        delete cuList[j];
}

void
Shader::updateContext(int cid) {
    // context of the thread which dispatched work
    assert(cpuPointer);
    gpuTc = cpuPointer->getContext(cid);
    assert(gpuTc);
}

void
Shader::execScheduledAdds()
{
    assert(!sa_when.empty());

    // apply any scheduled adds
    for (int i = 0; i < sa_n; ++i) {
        if (sa_when[i] <= curTick()) {
            *sa_val[i] += sa_x[i];
            panic_if(*sa_val[i] < 0, "Negative counter value\n");
            sa_val.erase(sa_val.begin() + i);
            sa_x.erase(sa_x.begin() + i);
            sa_when.erase(sa_when.begin() + i);
            --sa_n;
            --i;
        }
    }
    if (!sa_when.empty()) {
        Tick shader_wakeup = *std::max_element(sa_when.begin(),
                 sa_when.end());
        DPRINTF(GPUDisp, "Scheduling shader wakeup at %lu\n", shader_wakeup);
        schedule(tickEvent, shader_wakeup);
    } else {
        DPRINTF(GPUDisp, "sa_when empty, shader going to sleep!\n");
    }
}

/*
 * dispatcher/shader arranges invalidate requests to the CUs
 */
void
Shader::prepareInvalidate(HSAQueueEntry *task) {
    // if invalidate has already started/finished, then do nothing
    if (task->isInvStarted()) return;

    // invalidate has never started; it can only perform once at kernel launch
    assert(task->outstandingInvs() == -1);
    int kernId = task->dispatchId();
    // counter value is 0 now, indicating the inv is about to start
    _dispatcher.updateInvCounter(kernId, +1);

    // iterate all cus managed by the shader, to perform invalidate.
    for (int i_cu = 0; i_cu < n_cu; ++i_cu) {
        // create a request to hold INV info; the request's fields will
        // be updated in cu before use
        auto tcc_req = std::make_shared<Request>(0, 0, 0,
                                                 cuList[i_cu]->requestorId(),
                                                 0, -1);

        _dispatcher.updateInvCounter(kernId, +1);
        // all necessary INV flags are all set now, call cu to execute
        cuList[i_cu]->doInvalidate(tcc_req, task->dispatchId());


        // A set of CUs share a single SQC cache. Send a single invalidate
        // request to each SQC
        auto sqc_req = std::make_shared<Request>(0, 0, 0,
                                                 cuList[i_cu]->requestorId(),
                                                 0, -1);

        if ((i_cu % n_cu_per_sqc) == 0) {
            cuList[i_cu]->doSQCInvalidate(sqc_req, task->dispatchId());
        }

        // I don't like this. This is intrusive coding.
        cuList[i_cu]->resetRegisterPool();
    }
}

/**
 * dispatcher/shader arranges flush requests to the CUs
 */
void
Shader::prepareFlush(GPUDynInstPtr gpuDynInst){
    int kernId = gpuDynInst->kern_id;
    // flush has never been started, performed only once at kernel end
    assert(_dispatcher.getOutstandingWbs(kernId) == 0);

    // the first cu, managed by the shader, performs flush operation,
    // assuming that L2 cache is shared by all cus in the shader
    int i_cu = 0;
    _dispatcher.updateWbCounter(kernId, +1);
    cuList[i_cu]->doFlush(gpuDynInst);
}

bool
Shader::dispatchWorkgroups(HSAQueueEntry *task)
{
    bool scheduledSomething = false;
    int cuCount = 0;
    int curCu = nextSchedCu;
    int disp_count(0);

    while (cuCount < n_cu) {
        //Every time we try a CU, update nextSchedCu
        nextSchedCu = (nextSchedCu + 1) % n_cu;

        // dispatch workgroup iff the following two conditions are met:
        // (a) wg_rem is true - there are unassigned workgroups in the grid
        // (b) there are enough free slots in cu cuList[i] for this wg
        int num_wfs_in_wg = 0;
        bool can_disp = cuList[curCu]->hasDispResources(task, num_wfs_in_wg);
        if (!task->dispComplete() && can_disp) {
            scheduledSomething = true;
            DPRINTF(GPUDisp, "Dispatching a workgroup to CU %d: WG %d\n",
                            curCu, task->globalWgId());
            DPRINTF(GPUAgentDisp, "Dispatching a workgroup to CU %d: WG %d\n",
                            curCu, task->globalWgId());
            DPRINTF(GPUWgLatency, "WG Begin cycle:%d wg:%d cu:%d\n",
                    curTick(), task->globalWgId(), curCu);

            if (!cuList[curCu]->tickEvent.scheduled()) {
                if (!_activeCus)
                    _lastInactiveTick = curTick();
                _activeCus++;
            }

            panic_if(_activeCus <= 0 || _activeCus > cuList.size(),
                     "Invalid activeCu size\n");
            cuList[curCu]->dispWorkgroup(task, num_wfs_in_wg);

            task->markWgDispatch();
            ++disp_count;
        }

        ++cuCount;
        curCu = nextSchedCu;
    }

     DPRINTF(GPUWgLatency, "Shader Dispatched %d Wgs\n", disp_count);

    return scheduledSomething;
}

void
Shader::doFunctionalAccess(const RequestPtr &req, MemCmd cmd, void *data,
                           bool suppress_func_errors, int cu_id)
{
    int block_size = cuList.at(cu_id)->cacheLineSize();
    unsigned size = req->getSize();

    Addr tmp_addr;
    BaseMMU::Mode trans_mode;

    if (cmd == MemCmd::ReadReq) {
        trans_mode = BaseMMU::Read;
    } else if (cmd == MemCmd::WriteReq) {
        trans_mode = BaseMMU::Write;
    } else {
        fatal("unexcepted MemCmd\n");
    }

    tmp_addr = req->getVaddr();
    Addr split_addr = roundDown(tmp_addr + size - 1, block_size);

    assert(split_addr <= tmp_addr || split_addr - tmp_addr < block_size);

    // Misaligned access
    if (split_addr > tmp_addr) {
        RequestPtr req1, req2;
        req->splitOnVaddr(split_addr, req1, req2);

        PacketPtr pkt1 = new Packet(req2, cmd);
        PacketPtr pkt2 = new Packet(req1, cmd);

        functionalTLBAccess(pkt1, cu_id, trans_mode);
        functionalTLBAccess(pkt2, cu_id, trans_mode);

        PacketPtr new_pkt1 = new Packet(pkt1->req, cmd);
        PacketPtr new_pkt2 = new Packet(pkt2->req, cmd);

        new_pkt1->dataStatic(data);
        new_pkt2->dataStatic((uint8_t*)data + req1->getSize());

        if (suppress_func_errors) {
            new_pkt1->setSuppressFuncError();
            new_pkt2->setSuppressFuncError();
        }

        // fixme: this should be cuList[cu_id] if cu_id != n_cu
        // The latter requires a memPort in the dispatcher
        cuList[0]->memPort[0].sendFunctional(new_pkt1);
        cuList[0]->memPort[0].sendFunctional(new_pkt2);

        delete new_pkt1;
        delete new_pkt2;
        delete pkt1;
        delete pkt2;
    } else {
        PacketPtr pkt = new Packet(req, cmd);
        functionalTLBAccess(pkt, cu_id, trans_mode);
        PacketPtr new_pkt = new Packet(pkt->req, cmd);
        new_pkt->dataStatic(data);

        if (suppress_func_errors) {
            new_pkt->setSuppressFuncError();
        };

        // fixme: this should be cuList[cu_id] if cu_id != n_cu
        // The latter requires a memPort in the dispatcher
        cuList[0]->memPort[0].sendFunctional(new_pkt);

        delete new_pkt;
        delete pkt;
    }
}

void
Shader::ScheduleAdd(int *val,Tick when,int x)
{
    sa_val.push_back(val);
    when += curTick();
    sa_when.push_back(when);
    sa_x.push_back(x);
    ++sa_n;
    if (!tickEvent.scheduled() || (when < tickEvent.when())) {
        DPRINTF(GPUDisp, "New scheduled add; scheduling shader wakeup at "
                "%lu\n", when);
        reschedule(tickEvent, when, true);
    } else {
        assert(tickEvent.scheduled());
        DPRINTF(GPUDisp, "New scheduled add; wakeup already scheduled at "
                "%lu\n", when);
    }
}

void
Shader::AccessMem(uint64_t address, void *ptr, uint32_t size, int cu_id,
                  MemCmd cmd, bool suppress_func_errors)
{
    uint8_t *data_buf = (uint8_t*)ptr;

    for (ChunkGenerator gen(address, size, cuList.at(cu_id)->cacheLineSize());
         !gen.done(); gen.next()) {

        RequestPtr req = std::make_shared<Request>(
            gen.addr(), gen.size(), 0,
            cuList[0]->requestorId(), 0, 0, nullptr);

        doFunctionalAccess(req, cmd, data_buf, suppress_func_errors, cu_id);
        data_buf += gen.size();
    }
}

void
Shader::ReadMem(uint64_t address, void *ptr, uint32_t size, int cu_id)
{
    AccessMem(address, ptr, size, cu_id, MemCmd::ReadReq, false);
}

void
Shader::ReadMem(uint64_t address, void *ptr, uint32_t size, int cu_id,
                bool suppress_func_errors)
{
    AccessMem(address, ptr, size, cu_id, MemCmd::ReadReq,
        suppress_func_errors);
}

void
Shader::WriteMem(uint64_t address, void *ptr,uint32_t size, int cu_id)
{
    AccessMem(address, ptr, size, cu_id, MemCmd::WriteReq, false);
}

void
Shader::WriteMem(uint64_t address, void *ptr, uint32_t size, int cu_id,
                 bool suppress_func_errors)
{
    AccessMem(address, ptr, size, cu_id, MemCmd::WriteReq,
              suppress_func_errors);
}

/*
 * Send a packet through the appropriate TLB functional port.
 * If cu_id=n_cu, then this is the dispatcher's TLB.
 * Otherwise it's the TLB of the cu_id compute unit.
 */
void
Shader::functionalTLBAccess(PacketPtr pkt, int cu_id, BaseMMU::Mode mode)
{
    // update senderState. Need to know the gpuTc and the TLB mode
    pkt->senderState =
        new GpuTranslationState(mode, gpuTc, false);

    // even when the perLaneTLB flag is turned on
    // it's ok tp send all accesses through lane 0
    // since the lane # is not known here,
    // This isn't important since these are functional accesses.
    cuList[cu_id]->tlbPort[0].sendFunctional(pkt);

    /* safe_cast the senderState */
    GpuTranslationState *sender_state =
               safe_cast<GpuTranslationState*>(pkt->senderState);

    delete sender_state->tlbEntry;
    delete pkt->senderState;
}

/*
 * allow the shader to sample stats from constituent devices
 */
void
Shader::sampleStore(const Tick accessTime)
{
    stats.storeLatencyDist.sample(accessTime);
    stats.allLatencyDist.sample(accessTime);
}

/*
 * allow the shader to sample stats from constituent devices
 */
void
Shader::sampleLoad(const Tick accessTime)
{
    stats.loadLatencyDist.sample(accessTime);
    stats.allLatencyDist.sample(accessTime);
}

void
Shader::sampleInstRoundTrip(std::vector<Tick> roundTripTime)
{
    // Only sample instructions that go all the way to main memory
    if (roundTripTime.size() != InstMemoryHop::InstMemoryHopMax) {
        return;
    }

    Tick t1 = roundTripTime[0];
    Tick t2 = roundTripTime[1];
    Tick t3 = roundTripTime[2];
    Tick t4 = roundTripTime[3];
    Tick t5 = roundTripTime[4];

    stats.initToCoalesceLatency.sample(t2-t1);
    stats.rubyNetworkLatency.sample(t3-t2);
    stats.gmEnqueueLatency.sample(t4-t3);
    stats.gmToCompleteLatency.sample(t5-t4);
}

void
Shader::sampleLineRoundTrip(const std::map<Addr, std::vector<Tick>>& lineMap)
{
    stats.coalsrLineAddresses.sample(lineMap.size());
    std::vector<Tick> netTimes;

    // For each cache block address generated by a vmem inst, calculate
    // the round-trip time for that cache block.
    for (auto& it : lineMap) {
        const std::vector<Tick>& timeVec = it.second;
        if (timeVec.size() == 2) {
            netTimes.push_back(timeVec[1] - timeVec[0]);
        }
    }

    // Sort the cache block round trip times so that the first
    // distrubtion is always measuring the fastests and the last
    // distrubtion is always measuring the slowest cache block.
    std::sort(netTimes.begin(), netTimes.end());

    // Sample the round trip time for each N cache blocks into the
    // Nth distribution.
    int idx = 0;
    for (auto& time : netTimes) {
        stats.cacheBlockRoundTrip[idx].sample(time);
        ++idx;
    }
}

void
Shader::notifyCuSleep() {
    // If all CUs attached to his shader are asleep, update shaderActiveTicks
    panic_if(_activeCus <= 0 || _activeCus > cuList.size(),
             "Invalid activeCu size\n");
    _activeCus--;
    if (!_activeCus) {
        stats.shaderActiveTicks += curTick() - _lastInactiveTick;

        if (kernelExitRequested) {
            kernelExitRequested = false;
            if (blitKernel) {
                exitSimLoop("GPU Blit Kernel Completed");
            } else {
                exitSimLoop("GPU Kernel Completed");
            }
        }
    }
}

void
Shader::decNumOutstandingInvL2s()
{
    num_outstanding_invl2s--;

    if (num_outstanding_invl2s == 0 && !deferred_dispatches.empty()) {
        for (auto &dispatch : deferred_dispatches) {
            gpuCmdProc.submitDispatchPkt(std::get<0>(dispatch),
                                         std::get<1>(dispatch),
                                         std::get<2>(dispatch));
        }
        deferred_dispatches.clear();
    }
}

void
Shader::addDeferredDispatch(void *raw_pkt, uint32_t queue_id,
                            Addr host_pkt_addr)
{
    deferred_dispatches.push_back(
            std::make_tuple(raw_pkt, queue_id, host_pkt_addr));
}

/**
 * Forward the VRAM requestor ID needed for device memory from CP.
 */
RequestorID
Shader::vramRequestorId()
{
    return gpuCmdProc.vramRequestorId();
}

GfxVersion
Shader::getGfxVersion() const
{
    return gpuCmdProc.getGfxVersion();
}

Shader::ShaderStats::ShaderStats(statistics::Group *parent, int wf_size)
    : statistics::Group(parent),
      ADD_STAT(allLatencyDist, "delay distribution for all"),
      ADD_STAT(loadLatencyDist, "delay distribution for loads"),
      ADD_STAT(storeLatencyDist, "delay distribution for stores"),
      ADD_STAT(initToCoalesceLatency,
               "Ticks from vmem inst initiateAcc to coalescer issue"),
      ADD_STAT(rubyNetworkLatency,
               "Ticks from coalescer issue to coalescer hit callback"),
      ADD_STAT(gmEnqueueLatency,
               "Ticks from coalescer hit callback to GM pipe enqueue"),
      ADD_STAT(gmToCompleteLatency,
               "Ticks queued in GM pipes ordered response buffer"),
      ADD_STAT(coalsrLineAddresses,
               "Number of cache lines for coalesced request"),
      ADD_STAT(shaderActiveTicks,
               "Total ticks that any CU attached to this shader is active"),
      ADD_STAT(vectorInstSrcOperand,
               "vector instruction source operand distribution"),
      ADD_STAT(vectorInstDstOperand,
               "vector instruction destination operand distribution")
{
    allLatencyDist
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    loadLatencyDist
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    storeLatencyDist
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    initToCoalesceLatency
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    rubyNetworkLatency
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    gmEnqueueLatency
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    gmToCompleteLatency
        .init(0, 1600000-1, 10000)
        .flags(statistics::pdf | statistics::oneline);

    coalsrLineAddresses
        .init(0, 20, 1)
        .flags(statistics::pdf | statistics::oneline);

    vectorInstSrcOperand.init(4);
    vectorInstDstOperand.init(4);

    cacheBlockRoundTrip = new statistics::Distribution[wf_size];
    for (int idx = 0; idx < wf_size; ++idx) {
        std::stringstream namestr;
        ccprintf(namestr, "%s.cacheBlockRoundTrip%d",
                 static_cast<Shader*>(parent)->name(), idx);
        cacheBlockRoundTrip[idx]
            .init(0, 1600000-1, 10000)
            .name(namestr.str())
            .desc("Coalsr-to-coalsr time for the Nth cache block in an inst")
            .flags(statistics::pdf | statistics::oneline);
    }
}

void
Shader::startup()
{
    ClockedObject::startup();

    if (faultMask.empty())
        return;

    if (faultsScheduled)
        return;

    // Exclusive behavior: choose one model
    loadVrfFaultMask();
    if (!vrfFaults.empty()) {
        scheduleVrfFaults();
    } else {
        loadVrfHardFaultMask();
        if (!vrfHardFaults.empty()) {
            installVrfHardFaults();
        }
    }

    // --- SRF LOGIC (New) ---
    loadSrfFaultMask();
    if (!srfFaults.empty()) {
        scheduleSrfFaults();
    } else {
        loadSrfHardFaultMask();
        if (!srfHardFaults.empty()) {
            installSrfHardFaults();
        }
    }


loadSqcFaultMask();
if (!sqcFaults.empty()) {
    installSqcFaults();
    scheduleSqcFaults();
} else {
    loadSqcHardFaultMask();
    if (!sqcHardFaults.empty()) {
        installSqcHardFaults();
    }
}

loadTccFaultMask();
if (!tccFaults.empty()) {
    installTccFaults();
    scheduleTccFaults();
} else {
    loadTccHardFaultMask();
    if (!tccHardFaults.empty()) {
        installTccHardFaults();
    }
}
   loadTcpFaultMask();
if (!tcpFaults.empty()) {
    installTcpFaults();
    scheduleTcpFaults();
} else {
    loadTcpHardFaultMask();
    if (!tcpHardFaults.empty()) {
        installTcpHardFaults();
    }
}
    faultsScheduled = true;
}





void
Shader::scheduleSrfFaults()
{
    for (const auto& f : srfFaults) {
        InjectSrfFaultEvent *event = new InjectSrfFaultEvent(this, f);
        schedule(event, f.tick);
        inform("Scheduled SRF fault at tick %lu", f.tick);
    }
}


void
Shader::scheduleVrfFaults()
{
    for (const auto& f : vrfFaults) {
        // Use your specific class name here
        InjectVrfFaultEvent *event = new InjectVrfFaultEvent(this, f);

        schedule(event, f.tick);

        inform("Scheduled VRF fault at tick %lu", f.tick);
    }
}

void
Shader::loadSrfFaultMask()
{
    srfFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.srf_faults) {
        SrfFault f;

        // UNPACKING SRF TRANSIENT (matches apu_se.py packing)
        f.bit     =  packed        & 0x1F;   // 5 bits
        f.physReg = (packed >> 5)  & 0x7FF;  // 11 bits
        f.simd    = (packed >> 16) & 0x3;    // 2 bits
        f.cu      = (packed >> 18) & 0x3;    // 2 bits
        f.tick    =  packed >> 20;           // remaining bits

        srfFaults.push_back(f);

        inform("SUCCESS: Loaded SRF Fault: Tick=%lu CU=%d SIMD=%d Reg=%d Bit=%d\n",
               f.tick, f.cu, f.simd, f.physReg, f.bit);
    }
}

void
Shader::loadSrfHardFaultMask()
{
    srfHardFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.srf_hard_faults) {
        SrfHardFault f;

        f.stuckValue =  packed        & 0x1;    // 1 bit
        f.bit        = (packed >> 1)  & 0x1F;   // 5 bits
        f.physReg    = (packed >> 6)  & 0x7FF;  // 11 bits
        f.simd       = (packed >> 17) & 0x3;    // 2 bits
        f.cu         = (packed >> 19) & 0x3;    // 2 bits

        srfHardFaults.push_back(f);

        inform("SUCCESS: Loaded HARD SRF Fault: CU=%d SIMD=%d Reg=%d Bit=%d Stuck=%d\n",
               f.cu, f.simd, f.physReg, f.bit, f.stuckValue);
    }
}


void
Shader::loadVrfHardFaultMask()
{
    vrfHardFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.vrf_hard_faults) {
        VrfHardFault f;

        // UNPACK (matches apu_se.py hard packing):
        // [ cu(2) | simd(2) | reg(11) | lane(6) | bit(5) | stuck(1) ]
        f.stuckValue =  packed        & 0x1;   // 1 bit
        f.laneBit    = (packed >> 1)  & 0x1F;  // 5 bits
        f.lane       = (packed >> 6)  & 0x3F;  // 6 bits
        f.physReg    = (packed >> 12) & 0x7FF; // 11 bits
        f.simd       = (packed >> 23) & 0x3;   // 2 bits
        f.cu         = (packed >> 25) & 0x3;   // 2 bits

        vrfHardFaults.push_back(f);

        inform("SUCCESS: Loaded HARD Fault: CU=%d SIMD=%d Reg=%d Lane=%d Bit=%d Stuck=%d\n",
               f.cu, f.simd, f.physReg, f.lane, f.laneBit, f.stuckValue);
    }
}

  void
  Shader::installSrfHardFaults()
  {
      for (const auto &f : srfHardFaults) {
          auto cu = cuList[f.cu];

          // NOTE: rename 'srf' to the actual CU member if needed
          auto srf = cu->srf[f.simd];

          // You will implement these in ScalarRegisterFile next
          srf->addHardFault(f.physReg, f.bit, f.stuckValue);

          inform("Installed HARD SRF fault: CU=%d SIMD=%d Reg=%d Bit=%d Stuck=%d\n",
                 f.cu, f.simd, f.physReg, f.bit, f.stuckValue);
      }
  }



void
Shader::installVrfHardFaults()
{
    for (const auto &f : vrfHardFaults) {
        auto cu = cuList[f.cu];
        auto vrf = cu->vrf[f.simd];

        // This function does not exist yet — we’ll add it in VectorRegisterFile next
        vrf->addHardFault(f.physReg, f.lane, f.laneBit, f.stuckValue);

        inform("Installed HARD VRF fault: CU=%d SIMD=%d Reg=%d Lane=%d Bit=%d Stuck=%d\n",
               f.cu, f.simd, f.physReg, f.lane, f.laneBit, f.stuckValue);
    }
}

void
Shader::loadVrfFaultMask()
{
    vrfFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.vrf_faults) {
        VrfFault f;

        // UNPACKING (Matches the new 20 + 5 = 25 total shifts)
        f.laneBit =  packed        & 0x1F;   // 5 bits
        f.lane    = (packed >> 5)  & 0x3F;   // 6 bits (0..63)
        f.physReg = (packed >> 11) & 0x7FF;  // 11 bits (0..2047)
        f.simd    = (packed >> 22) & 0x3;    // 2 bits (0..3)
        f.cu      = (packed >> 24) & 0x3;    // 2 bits (0..3)
        f.tick    =  packed >> 26;


        vrfFaults.push_back(f);

        inform("SUCCESS: Loaded Fault: Tick=%lu CU=%d SIMD=%d Reg=%d Lane=%d Bit=%d\n",
               f.tick, f.cu, f.simd, f.physReg, f.lane, f.laneBit);
    }
}

void
gem5::Shader::InjectVrfFaultEvent::process()
{
    // 1. Get the target VRF
    auto cu = shader->cuList[fault.cu];
    auto vrf = cu->vrf[fault.simd];

    // 2. Trigger the injection in the VRF
    vrf->injectBitFlip(fault.physReg, fault.lane, fault.laneBit);
}
void
gem5::Shader::InjectSrfFaultEvent::process()
{
    auto cu = shader->cuList[fault.cu];

    // NOTE: rename 'srf' to the actual CU member if needed
    auto srf = cu->srf[fault.simd];

    srf->injectBitFlip(fault.physReg, fault.bit);
}
void Shader::loadTcpFaultMask() {
    tcpFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());
    for (auto packed : p.tcp_faults) {
        TcpFault f;
        // UNPACK: [ tick | cu(4) | set(4) | way(4) | byte(6) | bit(3) ]
        f.bit  =  packed        & 0x7;   // 3 bits
        f.byte = (packed >> 3)  & 0x3F;  // 6 bits
        f.way  = (packed >> 9)  & 0xF;   // 4 bits
        f.set  = (packed >> 13) & 0xF;   // 4 bits
        f.cu   = (packed >> 17) & 0xF;   // 4 bits
        f.tick =  packed >> 21;          // remaining bits
        tcpFaults.push_back(f);
        inform("Loaded TCP Fault: Tick=%lu CU=%d Set=%d Way=%d Byte=%d Bit=%d\n",
               f.tick, f.cu, f.set, f.way, f.byte, f.bit);
    }
}

void Shader::scheduleTcpFaults() {
    for (const auto& f : tcpFaults) {
        schedule(new InjectTcpFaultEvent(this, f), f.tick);
    }
}

void
Shader::InjectTcpFaultEvent::process()
{
    // Inside this scope, 'shader' and 'fault' are member
    // variables of the InjectTcpFaultEvent class.
    auto cu = shader->cuList[fault.cu];

    // Call the injection on the ComputeUnit
    cu->injectTcpBitFlip(fault.set, fault.way, fault.byte, fault.bit);
}

void Shader::loadTcpHardFaultMask() {
    tcpHardFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());
    for (auto packed : p.tcp_hard_faults) {
        TcpHardFault f;
        // UNPACK: [ cu(4) | set(4) | way(4) | byte(6) | bit(3) | stuck(1) ]
        f.stuckValue =  packed        & 0x1;
        f.bit        = (packed >> 1)  & 0x7;
        f.byte       = (packed >> 4)  & 0x3F;
        f.way        = (packed >> 10) & 0xF;
        f.set        = (packed >> 14) & 0xF;
        f.cu         = (packed >> 18) & 0xF;
        tcpHardFaults.push_back(f);
    }
}

void Shader::installTcpHardFaults() {
    for (const auto &f : tcpHardFaults) {
        inform("Shader: Forwarding HARD TCP fault to CU=%d Set=%d Way=%d Byte=%d Bit=%d Stuck=%d",
               f.cu, f.set, f.way, f.byte, f.bit, f.stuckValue);

               inform("FAULT_TRACE event=TCP_SOURCE fault_id=0 tick=%llu "
       "component=TCP source_cu=%d set=%d way=%d "
       "byte=%d bit=%d stuck=%d",
       static_cast<unsigned long long>(curTick()),
       f.cu,
       f.set,
       f.way,
       f.byte,
       f.bit,
       f.stuckValue);

        // Check if the Compute Unit actually exists, then send it down the bridge!
        if (f.cu < n_cu && cuList[f.cu]) {
            cuList[f.cu]->loadTcpHardFault(f.set, f.way, f.byte, f.bit, f.stuckValue);
        } else {
            warn("Shader: Cannot install fault! CU %d is out of bounds or null.", f.cu);
        }
    }
}




void Shader::loadSqcFaultMask()
{
    sqcFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.sqc_faults) {
        SqcFault f;

        f.bit  =  packed        & 0x7;    // 3 bits
        f.byte = (packed >> 3)  & 0x3F;   // 6 bits
        f.way  = (packed >> 9)  & 0x7;    // 3 bits
        f.set  = (packed >> 12) & 0x3F;   // 6 bits
        f.sqc  = (packed >> 18) & 0xF;    // 4 bits
        f.tick =  packed >> 22;

        sqcFaults.push_back(f);

        inform("Loaded SQC Fault: Tick=%lu SQC=%d Set=%d Way=%d Byte=%d Bit=%d",
               f.tick, f.sqc, f.set, f.way, f.byte, f.bit);
    }
}
void
gem5::Shader::InjectSqcFaultEvent::process()
{
    int base_cu = fault.sqc * shader->n_cu_per_sqc;

    if (base_cu >= shader->cuList.size() || !shader->cuList[base_cu]) {
        warn("SQC transient fault event cannot find base CU for SQC=%d", fault.sqc);
        return;
    }

    shader->cuList[base_cu]->injectSqcBitFlip(
        fault.sqc, fault.set, fault.way, fault.byte, fault.bit);
}

void Shader::scheduleSqcFaults()
{
    for (const auto& f : sqcFaults) {
        schedule(new InjectSqcFaultEvent(this, f), f.tick);
    }
}

void Shader::loadSqcHardFaultMask()
{
    sqcHardFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.sqc_hard_faults) {
        SqcHardFault f;

        f.stuckValue =  packed        & 0x1;
        f.bit        = (packed >> 1)  & 0x7;
        f.byte       = (packed >> 4)  & 0x3F;
        f.way        = (packed >> 10) & 0x7;
        f.set        = (packed >> 13) & 0x3F;
        f.sqc        = (packed >> 19) & 0xF;

        sqcHardFaults.push_back(f);

        inform("Loaded HARD SQC Fault: SQC=%d Set=%d Way=%d Byte=%d Bit=%d Stuck=%d",
               f.sqc, f.set, f.way, f.byte, f.bit, f.stuckValue);
    }
}

void Shader::installSqcHardFaults()
{
    for (const auto &f : sqcHardFaults) {
        int base_cu = f.sqc * n_cu_per_sqc;

        inform("Shader: Forwarding HARD SQC fault to SQC=%d via CU=%d Set=%d Way=%d Byte=%d Bit=%d Stuck=%d",
               f.sqc, base_cu, f.set, f.way, f.byte, f.bit, f.stuckValue);

        if (base_cu < n_cu && cuList[base_cu]) {
            cuList[base_cu]->loadSqcHardFault(
                f.sqc, f.set, f.way, f.byte, f.bit, f.stuckValue);
        } else {
            warn("Shader: Cannot install HARD SQC fault! base CU %d invalid for SQC %d",
                 base_cu, f.sqc);
        }
    }
}

void Shader::installSqcFaults()
{
    for (const auto &f : sqcFaults) {
        int base_cu = f.sqc * n_cu_per_sqc;

        inform("Shader: Forwarding TRANSIENT SQC fault to SQC=%d via CU=%d Set=%d Way=%d Byte=%d Bit=%d Tick=%lu",
               f.sqc, base_cu, f.set, f.way, f.byte, f.bit, f.tick);

        if (base_cu < n_cu && cuList[base_cu]) {
            cuList[base_cu]->loadSqcTransientFault(
                f.sqc, f.set, f.way, f.byte, f.bit);
        } else {
            warn("Shader: Cannot install transient SQC fault! base CU %d invalid for SQC %d",
                 base_cu, f.sqc);
        }
    }
}

void Shader::installTcpFaults()
{
    for (const auto &f : tcpFaults) {
        inform("Shader: Forwarding TRANSIENT TCP fault to CU=%d Set=%d Way=%d Byte=%d Bit=%d Tick=%lu",
               f.cu, f.set, f.way, f.byte, f.bit, f.tick);

        if (f.cu < n_cu && cuList[f.cu]) {
            cuList[f.cu]->loadTcpTransientFault(f.set, f.way, f.byte, f.bit);
        } else {
            warn("Shader: Cannot install transient TCP fault! CU %d is out of bounds or null.", f.cu);
        }
    }
}

void Shader::scheduleTccFaults()
{
    for (const auto &f : tccFaults) {
        schedule(new InjectTccFaultEvent(this, f), f.tick);
    }
}

void
Shader::injectTccBitFlip(int tcc, int set, int way, int byte, int bit)
{
    if (tcc < 0 || tcc >= tccCaches.size() || tccCaches[tcc] == nullptr) {
        warn("Shader: invalid TCC transient target tcc=%d", tcc);
        return;
    }

    bool flipped = tccCaches[tcc]->activateTccTransientFaultIfMatch(set, way);

    inform("Shader: TCC transient injection tcc=%d set=%d way=%d "
           "byte=%d bit=%d flipped=%d",
           tcc, set, way, byte, bit, flipped);
}



void
gem5::Shader::InjectTccFaultEvent::process()
{
    shader->injectTccBitFlip(fault.tcc, fault.set, fault.way, fault.byte, fault.bit);
}

void Shader::loadTccFaultMask()
{
    tccFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.tcc_faults) {
        TccFault f;

        f.bit  =  packed        & 0x7;
        f.byte = (packed >> 3)  & 0x3F;
        f.way  = (packed >> 9)  & 0xF;
        f.set  = (packed >> 13) & 0xFF;
        f.tcc  = (packed >> 21) & 0xF;
        f.tick =  packed >> 25;

        tccFaults.push_back(f);

        inform("Loaded TCC Fault: Tick=%lu TCC=%d Set=%d Way=%d Byte=%d Bit=%d\n",
               f.tick, f.tcc, f.set, f.way, f.byte, f.bit);
    }
}

void Shader::loadTccHardFaultMask()
{
    tccHardFaults.clear();
    const ShaderParams &p = reinterpret_cast<const ShaderParams &>(params());

    for (auto packed : p.tcc_hard_faults) {
        TccHardFault f;

        f.stuckValue =  packed        & 0x1;
        f.bit        = (packed >> 1)  & 0x7;
        f.byte       = (packed >> 4)  & 0x3F;
        f.way        = (packed >> 10) & 0xF;
        f.set        = (packed >> 14) & 0xFF;
        f.tcc        = (packed >> 22) & 0xF;

        tccHardFaults.push_back(f);

        inform("Loaded HARD TCC Fault: TCC=%d Set=%d Way=%d Byte=%d Bit=%d Stuck=%d\n",
               f.tcc, f.set, f.way, f.byte, f.bit, f.stuckValue);
    }
}

void
Shader::installTccFaults()
{
    for (const auto &f : tccFaults) {
        inform("Shader: Preparing TRANSIENT TCC fault TCC=%d Set=%d Way=%d "
               "Byte=%d Bit=%d Tick=%lu",
               f.tcc, f.set, f.way, f.byte, f.bit, f.tick);

        if (f.tcc < 0 || f.tcc >= tccCaches.size() || tccCaches[f.tcc] == nullptr) {
            warn("Shader: invalid TCC transient preload target tcc=%d", f.tcc);
            continue;
        }

        tccCaches[f.tcc]->loadTccTransientFault(
            f.set, f.way, f.byte, f.bit
        );
    }
}

void
Shader::loadTccHardFault(int tcc, int set, int way, int byte, int bit,
                         int stuckValue)
{
    if (tcc < 0 || tcc >= tccCaches.size() || tccCaches[tcc] == nullptr) {
        warn("Shader: invalid TCC hard fault target tcc=%d", tcc);
        return;
    }

    tccCaches[tcc]->loadTccHardFault(set, way, byte, bit, stuckValue);

    inform("Shader: forwarded HARD TCC fault tcc=%d set=%d way=%d byte=%d "
           "bit=%d stuck=%d",
           tcc, set, way, byte, bit, stuckValue);
}

void Shader::installTccHardFaults()
{
    for (const auto &f : tccHardFaults) {
        inform("Shader: Forwarding HARD TCC fault TCC=%d Set=%d Way=%d Byte=%d Bit=%d Stuck=%d",
               f.tcc, f.set, f.way, f.byte, f.bit, f.stuckValue);

        loadTccHardFault(f.tcc, f.set, f.way, f.byte, f.bit, f.stuckValue);
    }
}




void
Shader::addTccPoison(Addr addr)
{
    tccPoisonedLines.insert(addr);
}

void
Shader::removeTccPoison(Addr addr)
{
    tccPoisonedLines.erase(addr);
}

bool
Shader::isTccPoisoned(Addr addr) const
{
    return tccPoisonedLines.count(addr) != 0;
}


} // namespace gem5
