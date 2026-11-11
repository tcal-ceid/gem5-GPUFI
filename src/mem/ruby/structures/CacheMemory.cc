/*
 * Copyright (c) 2020-2021 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Copyright (c) 1999-2012 Mark D. Hill and David A. Wood
 * Copyright (c) 2013 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "mem/ruby/structures/CacheMemory.hh"

#include "base/compiler.hh"
#include "base/intmath.hh"
#include "base/logging.hh"
#include "debug/HtmMem.hh"
#include "debug/RubyCache.hh"
#include "gpu-compute/shader.hh"
#include "debug/RubyCacheTrace.hh"
#include "debug/RubyResourceStalls.hh"
#include "debug/RubyStats.hh"
#include "mem/cache/replacement_policies/weighted_lru_rp.hh"
#include "mem/ruby/protocol/AccessPermission.hh"
#include "mem/ruby/system/RubySystem.hh"
#include <string>

namespace gem5
{

namespace ruby
{

std::ostream&
operator<<(std::ostream& out, const CacheMemory& obj)
{
    obj.print(out);
    out << std::flush;
    return out;
}

CacheMemory::CacheMemory(const Params &p)
    : SimObject(p),
    dataArray(p.dataArrayBanks, p.dataAccessLatency, p.start_index_bit),
    tagArray(p.tagArrayBanks, p.tagAccessLatency, p.start_index_bit),
    atomicALUArray(p.atomicALUs, p.atomicLatency),
    cacheMemoryStats(this)
{
    m_cache_size = p.size;
    m_cache_assoc = p.assoc;
    m_replacementPolicy_ptr = p.replacement_policy;
    m_start_index_bit = p.start_index_bit;
    m_is_instruction_only_cache = p.is_icache;
    m_resource_stalls = p.resourceStalls;
    m_block_size = p.block_size;  // may be 0 at this point. Updated in init()
    m_use_occupancy = dynamic_cast<replacement_policy::WeightedLRU*>(
                                    m_replacementPolicy_ptr) ? true : false;
}

std::unordered_set<Addr> CacheMemory::tcc_poisoned_lines;

void
CacheMemory::setRubySystem(RubySystem* rs)
{
    dataArray.setClockPeriod(rs->clockPeriod());
    tagArray.setClockPeriod(rs->clockPeriod());
    atomicALUArray.setClockPeriod(rs->clockPeriod());
    atomicALUArray.setBlockSize(rs->getBlockSizeBytes());

    if (m_block_size == 0) {
        m_block_size = rs->getBlockSizeBytes();
    }

    m_ruby_system = rs;
}

void
CacheMemory::init()
{
    assert(m_block_size != 0);
    m_cache_num_sets = (m_cache_size / m_cache_assoc) / m_block_size;
    assert(m_cache_num_sets > 1);
    m_cache_num_set_bits = floorLog2(m_cache_num_sets);
    assert(m_cache_num_set_bits > 0);

    m_cache.resize(m_cache_num_sets,
                    std::vector<AbstractCacheEntry*>(m_cache_assoc, nullptr));
    replacement_data.resize(m_cache_num_sets,
                               std::vector<ReplData>(m_cache_assoc, nullptr));
    // instantiate all the replacement_data here
    for (int i = 0; i < m_cache_num_sets; i++) {
        for ( int j = 0; j < m_cache_assoc; j++) {
            replacement_data[i][j] =
                                m_replacementPolicy_ptr->instantiateEntry();
        }
    }
}

CacheMemory::~CacheMemory()
{
    if (m_replacementPolicy_ptr)
        delete m_replacementPolicy_ptr;
    for (int i = 0; i < m_cache_num_sets; i++) {
        for (int j = 0; j < m_cache_assoc; j++) {
            delete m_cache[i][j];
        }
    }
}

// convert a Address to its location in the cache
int64_t
CacheMemory::addressToCacheSet(Addr address) const
{
    assert(address == makeLineAddress(address));
    return bitSelect(address, m_start_index_bit,
                     m_start_index_bit + m_cache_num_set_bits - 1);
}

// Given a cache index: returns the index of the tag in a set.
// returns -1 if the tag is not found.
int
CacheMemory::findTagInSet(int64_t cacheSet, Addr tag) const
{
    assert(tag == makeLineAddress(tag));
    // search the set for the tags
    auto it = m_tag_index.find(tag);
    if (it != m_tag_index.end())
        if (m_cache[cacheSet][it->second]->m_Permission !=
            AccessPermission_NotPresent)
            return it->second;
    return -1; // Not found
}

// Given a cache index: returns the index of the tag in a set.
// returns -1 if the tag is not found.
int
CacheMemory::findTagInSetIgnorePermissions(int64_t cacheSet,
                                           Addr tag) const
{
    assert(tag == makeLineAddress(tag));
    // search the set for the tags
    auto it = m_tag_index.find(tag);
    if (it != m_tag_index.end())
        return it->second;
    return -1; // Not found
}

// Given an unique cache block identifier (idx): return the valid address
// stored by the cache block.  If the block is invalid/notpresent, the
// function returns the 0 address
Addr
CacheMemory::getAddressAtIdx(int idx) const
{
    Addr tmp(0);

    int set = idx / m_cache_assoc;
    assert(set < m_cache_num_sets);

    int way = idx - set * m_cache_assoc;
    assert (way < m_cache_assoc);

    AbstractCacheEntry* entry = m_cache[set][way];
    if (entry == NULL ||
        entry->m_Permission == AccessPermission_Invalid ||
        entry->m_Permission == AccessPermission_NotPresent) {
        return tmp;
    }
    return entry->m_Address;
}

bool
CacheMemory::tryCacheAccess(Addr address, RubyRequestType type,
                            DataBlock*& data_ptr)
{
    DPRINTF(RubyCache, "trying to access address: %#x\n", address);
    AbstractCacheEntry* entry = lookup(address);
    if (entry != nullptr) {
        // Do we even have a tag match?
        m_replacementPolicy_ptr->touch(entry->replacementData);
        entry->setLastAccess(curTick());
        data_ptr = &(entry->getDataBlk());

        if (entry->m_Permission == AccessPermission_Read_Write) {
            DPRINTF(RubyCache, "Have permission to access address: %#x\n",
                        address);
            return true;
        }
        if ((entry->m_Permission == AccessPermission_Read_Only) &&
            (type == RubyRequestType_LD || type == RubyRequestType_IFETCH)) {
            DPRINTF(RubyCache, "Have permission to access address: %#x\n",
                        address);
            return true;
        }
        // The line must not be accessible
    }
    DPRINTF(RubyCache, "Do not have permission to access address: %#x\n",
                address);
    data_ptr = NULL;
    return false;
}

bool
CacheMemory::testCacheAccess(Addr address, RubyRequestType type,
                             DataBlock*& data_ptr)
{
    DPRINTF(RubyCache, "testing address: %#x\n", address);
    AbstractCacheEntry* entry = lookup(address);
    if (entry != nullptr) {
        // Do we even have a tag match?
        m_replacementPolicy_ptr->touch(entry->replacementData);
        entry->setLastAccess(curTick());
        data_ptr = &(entry->getDataBlk());

        DPRINTF(RubyCache, "have permission for address %#x?: %d\n",
                    address,
                    entry->m_Permission != AccessPermission_NotPresent);
        return entry->m_Permission != AccessPermission_NotPresent;
    }

    DPRINTF(RubyCache, "do not have permission for address %#x\n",
                address);
    data_ptr = NULL;
    return false;
}

// tests to see if an address is present in the cache
bool
CacheMemory::isTagPresent(Addr address) const
{
    const AbstractCacheEntry* const entry = lookup(address);
    if (entry == nullptr) {
        // We didn't find the tag
        DPRINTF(RubyCache, "No tag match for address: %#x\n", address);
        return false;
    }
    DPRINTF(RubyCache, "address: %#x found\n", address);
    return true;
}

// Returns true if there is:
//   a) a tag match on this address or there is
//   b) an unused line in the same cache "way"
bool
CacheMemory::cacheAvail(Addr address) const
{
    assert(address == makeLineAddress(address));

    int64_t cacheSet = addressToCacheSet(address);

    for (int i = 0; i < m_cache_assoc; i++) {
        AbstractCacheEntry* entry = m_cache[cacheSet][i];
        if (entry != NULL) {
            if (entry->m_Address == address ||
                entry->m_Permission == AccessPermission_NotPresent) {
                // Already in the cache or we found an empty entry
                return true;
            }
        } else {
            return true;
        }
    }
    return false;
}

AbstractCacheEntry*
CacheMemory::allocate(Addr address, AbstractCacheEntry *entry)
{
    assert(address == makeLineAddress(address));
    assert(!isTagPresent(address));
    assert(cacheAvail(address));
    DPRINTF(RubyCache, "allocating address: %#x\n", address);

    entry->initBlockSize(m_block_size);
    entry->setRubySystem(m_ruby_system);

    // Find the first open slot
    int64_t cacheSet = addressToCacheSet(address);
    std::vector<AbstractCacheEntry*> &set = m_cache[cacheSet];
    for (int i = 0; i < m_cache_assoc; i++) {
        if (!set[i] || set[i]->m_Permission == AccessPermission_NotPresent) {

            // --- NEW: TRANSIENT FAULT ERASER (OVERWRITE/REFILL) ---
            // '0' acts as a wildcard address since the old data is gone
            clearTcpTransientFaultIfMatch(cacheSet, i, 0);
            clearSqcTransientFaultIfMatch(cacheSet, i, 0);
            clearTccTransientFaultIfMatch(cacheSet, i, 0);
            // ------------------------------------------------------

            if (set[i] && (set[i] != entry)) {
                warn_once("This protocol contains a cache entry handling bug: "
                    "Entries in the cache should never be NotPresent! If\n"
                    "this entry (%#x) is not tracked elsewhere, it will memory "
                    "leak here. Fix your protocol to eliminate these!",
                    address);
            }
            set[i] = entry;  // Init entry
            set[i]->m_Address = address;
            set[i]->m_Permission = AccessPermission_Invalid;
            DPRINTF(RubyCache, "Allocate clearing lock for addr: 0x%x\n",
                    address);
            set[i]->m_locked = -1;
            m_tag_index[address] = i;
            set[i]->setPosition(cacheSet, i);
            set[i]->replacementData = replacement_data[cacheSet][i];
            set[i]->setLastAccess(curTick());

            // Call reset function here to set initial value for different
            // replacement policies.
            m_replacementPolicy_ptr->reset(entry->replacementData);

            return entry;
        }
    }
    panic("Allocate didn't find an available entry");
}
void
CacheMemory::deallocate(Addr address)
{
   if (name().find("tcc_cntrl") != std::string::npos) {
        CacheMemory::removeTccPoison(makeLineAddress(address));
    }
    AbstractCacheEntry* entry = lookup(address);
    assert(entry != nullptr);
    m_replacementPolicy_ptr->invalidate(entry->replacementData);
    uint32_t cache_set = entry->getSet();
    uint32_t way = entry->getWay();

    clearTcpTransientFaultIfMatch(cache_set, way, address);
    clearSqcTransientFaultIfMatch(cache_set, way, address);
    clearTccTransientFaultIfMatch(cache_set, way, address);

    delete entry;
    m_cache[cache_set][way] = NULL;
    m_tag_index.erase(address);
}

// Returns with the physical address of the conflicting cache line
Addr
CacheMemory::cacheProbe(Addr address) const
{
    assert(address == makeLineAddress(address));
    assert(!cacheAvail(address));

    int64_t cacheSet = addressToCacheSet(address);
    std::vector<ReplaceableEntry*> candidates;
    for (int i = 0; i < m_cache_assoc; i++) {
        candidates.push_back(static_cast<ReplaceableEntry*>(
                                                       m_cache[cacheSet][i]));
    }
    return m_cache[cacheSet][m_replacementPolicy_ptr->
                        getVictim(candidates)->getWay()]->m_Address;
}

// looks an address up in the cache
AbstractCacheEntry*
CacheMemory::lookup(Addr address)
{
    assert(address == makeLineAddress(address));
    int64_t cacheSet = addressToCacheSet(address);
    int loc = findTagInSet(cacheSet, address);

    if (loc == -1) {
        return NULL;
    }

// --- NEW: TRIGGER THE L2 TCC FAULT ---
    if (name().find("tcc_cntrl") != std::string::npos) {
        enforceTccHardFaultsOnLine(cacheSet, loc);
        // REMOVED activateTccTransientFaultIfMatch HERE!
    }

    return m_cache[cacheSet][loc];
}
const AbstractCacheEntry*
CacheMemory::lookup(Addr address) const
{
    assert(address == makeLineAddress(address));
    int64_t cacheSet = addressToCacheSet(address);
    int loc = findTagInSet(cacheSet, address);
    if (loc == -1) {
        return NULL;
    }

    return m_cache[cacheSet][loc];
}
// Sets the most recently used bit for a cache block
void
CacheMemory::setMRU(Addr address)
{
    AbstractCacheEntry* entry = lookup(makeLineAddress(address));
    if (entry != nullptr) {
        m_replacementPolicy_ptr->touch(entry->replacementData);
        entry->setLastAccess(curTick());
    }
}

void
CacheMemory::setMRU(AbstractCacheEntry *entry)
{
    assert(entry != nullptr);
    m_replacementPolicy_ptr->touch(entry->replacementData);
    entry->setLastAccess(curTick());
}

void
CacheMemory::setMRU(Addr address, int occupancy)
{
    AbstractCacheEntry* entry = lookup(makeLineAddress(address));
    if (entry != nullptr) {
        // m_use_occupancy can decide whether we are using WeightedLRU
        // replacement policy. Depending on different replacement policies,
        // use different touch() function.
        if (m_use_occupancy) {
            static_cast<replacement_policy::WeightedLRU*>(
                m_replacementPolicy_ptr)->touch(
                entry->replacementData, occupancy);
        } else {
            m_replacementPolicy_ptr->touch(entry->replacementData);
        }
        entry->setLastAccess(curTick());
    }
}

int
CacheMemory::getReplacementWeight(int64_t set, int64_t loc)
{
    assert(set < m_cache_num_sets);
    assert(loc < m_cache_assoc);
    int ret = 0;
    if (m_cache[set][loc] != NULL) {
        ret = m_cache[set][loc]->getNumValidBlocks();
        assert(ret >= 0);
    }

    return ret;
}

void
CacheMemory::recordCacheContents(int cntrl, CacheRecorder* tr) const
{
    uint64_t warmedUpBlocks = 0;
    [[maybe_unused]] uint64_t totalBlocks = (uint64_t)m_cache_num_sets *
                                         (uint64_t)m_cache_assoc;

    for (int i = 0; i < m_cache_num_sets; i++) {
        for (int j = 0; j < m_cache_assoc; j++) {
            if (m_cache[i][j] != NULL) {
                AccessPermission perm = m_cache[i][j]->m_Permission;
                RubyRequestType request_type = RubyRequestType_NULL;
                if (perm == AccessPermission_Read_Only) {
                    if (m_is_instruction_only_cache) {
                        request_type = RubyRequestType_IFETCH;
                    } else {
                        request_type = RubyRequestType_LD;
                    }
                } else if (perm == AccessPermission_Read_Write) {
                    request_type = RubyRequestType_ST;
                }

                if (request_type != RubyRequestType_NULL) {
                    Tick lastAccessTick;
                    lastAccessTick = m_cache[i][j]->getLastAccess();
                    tr->addRecord(cntrl, m_cache[i][j]->m_Address,
                                  0, request_type, lastAccessTick,
                                  m_cache[i][j]->getDataBlk());
                    warmedUpBlocks++;
                }
            }
        }
    }

    DPRINTF(RubyCacheTrace, "%s: %lli blocks of %lli total blocks"
            "recorded %.2f%% \n", name().c_str(), warmedUpBlocks,
            totalBlocks, (float(warmedUpBlocks) / float(totalBlocks)) * 100.0);
}

void
CacheMemory::print(std::ostream& out) const
{
    out << "Cache dump: " << name() << std::endl;
    for (int i = 0; i < m_cache_num_sets; i++) {
        for (int j = 0; j < m_cache_assoc; j++) {
            if (m_cache[i][j] != NULL) {
                out << "  Index: " << i
                    << " way: " << j
                    << " entry: " << *m_cache[i][j] << std::endl;
            } else {
                out << "  Index: " << i
                    << " way: " << j
                    << " entry: NULL" << std::endl;
            }
        }
    }
}

void
CacheMemory::printData(std::ostream& out) const
{
    out << "printData() not supported" << std::endl;
}

void
CacheMemory::setLocked(Addr address, int context)
{
    DPRINTF(RubyCache, "Setting Lock for addr: %#x to %d\n", address, context);
    AbstractCacheEntry* entry = lookup(address);
    assert(entry != nullptr);
    entry->setLocked(context);
}

void
CacheMemory::clearLocked(Addr address)
{
    DPRINTF(RubyCache, "Clear Lock for addr: %#x\n", address);
    AbstractCacheEntry* entry = lookup(address);
    assert(entry != nullptr);
    entry->clearLocked();
}

void
CacheMemory::clearLockedAll(int context)
{
    // iterate through every set and way to get a cache line
    for (auto i = m_cache.begin(); i != m_cache.end(); ++i) {
        std::vector<AbstractCacheEntry*> set = *i;
        for (auto j = set.begin(); j != set.end(); ++j) {
            AbstractCacheEntry *line = *j;
            if (line && line->isLocked(context)) {
                DPRINTF(RubyCache, "Clear Lock for addr: %#x\n",
                    line->m_Address);
                line->clearLocked();
            }
        }
    }
}

bool
CacheMemory::isLocked(Addr address, int context)
{
    AbstractCacheEntry* entry = lookup(address);
    assert(entry != nullptr);
    DPRINTF(RubyCache, "Testing Lock for addr: %#llx cur %d con %d\n",
            address, entry->m_locked, context);
    return entry->isLocked(context);
}

CacheMemory::
CacheMemoryStats::CacheMemoryStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(numDataArrayReads, "Number of data array reads"),
      ADD_STAT(numDataArrayWrites, "Number of data array writes"),
      ADD_STAT(numTagArrayReads, "Number of tag array reads"),
      ADD_STAT(numTagArrayWrites, "Number of tag array writes"),
      ADD_STAT(numTagArrayStalls, "Number of stalls caused by tag array"),
      ADD_STAT(numDataArrayStalls, "Number of stalls caused by data array"),
      ADD_STAT(numAtomicALUOperations, "Number of atomic ALU operations"),
      ADD_STAT(numAtomicALUArrayStalls, "Number of stalls caused by atomic ALU array"),
      ADD_STAT(htmTransCommitReadSet, "Read set size of a committed "
                                      "transaction"),
      ADD_STAT(htmTransCommitWriteSet, "Write set size of a committed "
                                       "transaction"),
      ADD_STAT(htmTransAbortReadSet, "Read set size of a aborted transaction"),
      ADD_STAT(htmTransAbortWriteSet, "Write set size of a aborted "
                                      "transaction"),
      ADD_STAT(m_demand_hits, "Number of cache demand hits"),
      ADD_STAT(m_demand_misses, "Number of cache demand misses"),
      ADD_STAT(m_demand_accesses, "Number of cache demand accesses",
               m_demand_hits + m_demand_misses),
      ADD_STAT(m_prefetch_hits, "Number of cache prefetch hits"),
      ADD_STAT(m_prefetch_misses, "Number of cache prefetch misses"),
      ADD_STAT(m_prefetch_accesses, "Number of cache prefetch accesses",
               m_prefetch_hits + m_prefetch_misses),
      ADD_STAT(m_accessModeType, "")
{
    numDataArrayReads
        .flags(statistics::nozero);

    numDataArrayWrites
        .flags(statistics::nozero);

    numTagArrayReads
        .flags(statistics::nozero);

    numTagArrayWrites
        .flags(statistics::nozero);

    numTagArrayStalls
        .flags(statistics::nozero);

    numDataArrayStalls
        .flags(statistics::nozero);

    numAtomicALUOperations
        .flags(statistics::nozero);

    numAtomicALUArrayStalls
        .flags(statistics::nozero);

    htmTransCommitReadSet
        .init(8)
        .flags(statistics::pdf | statistics::dist | statistics::nozero |
            statistics::nonan);

    htmTransCommitWriteSet
        .init(8)
        .flags(statistics::pdf | statistics::dist | statistics::nozero |
            statistics::nonan);

    htmTransAbortReadSet
        .init(8)
        .flags(statistics::pdf | statistics::dist | statistics::nozero |
            statistics::nonan);

    htmTransAbortWriteSet
        .init(8)
        .flags(statistics::pdf | statistics::dist | statistics::nozero |
            statistics::nonan);

    m_prefetch_hits
        .flags(statistics::nozero);

    m_prefetch_misses
        .flags(statistics::nozero);

    m_prefetch_accesses
        .flags(statistics::nozero);

    m_accessModeType
        .init(RubyRequestType_NUM)
        .flags(statistics::pdf | statistics::total);

    for (int i = 0; i < RubyAccessMode_NUM; i++) {
        m_accessModeType
            .subname(i, RubyAccessMode_to_string(RubyAccessMode(i)))
            .flags(statistics::nozero)
            ;
    }
}

// assumption: SLICC generated files will only call this function
// once **all** resources are granted
void
CacheMemory::recordRequestType(CacheRequestType requestType, Addr addr)
{
    DPRINTF(RubyStats, "Recorded statistic: %s\n",
            CacheRequestType_to_string(requestType));
    switch(requestType) {
    case CacheRequestType_DataArrayRead:
        if (m_resource_stalls)
            dataArray.reserve(addressToCacheSet(addr));
        cacheMemoryStats.numDataArrayReads++;
        return;
    case CacheRequestType_DataArrayWrite:
        if (m_resource_stalls)
            dataArray.reserve(addressToCacheSet(addr));
        cacheMemoryStats.numDataArrayWrites++;
        return;
    case CacheRequestType_TagArrayRead:
        if (m_resource_stalls)
            tagArray.reserve(addressToCacheSet(addr));
        cacheMemoryStats.numTagArrayReads++;
        return;
    case CacheRequestType_TagArrayWrite:
        if (m_resource_stalls)
            tagArray.reserve(addressToCacheSet(addr));
        cacheMemoryStats.numTagArrayWrites++;
        return;
    case CacheRequestType_AtomicALUOperation:
        if (m_resource_stalls)
            atomicALUArray.reserve(addr);
        cacheMemoryStats.numAtomicALUOperations++;
        return;
    default:
        warn("CacheMemory access_type not found: %s",
             CacheRequestType_to_string(requestType));
    }
}

bool
CacheMemory::checkResourceAvailable(CacheResourceType res, Addr addr)
{
    if (!m_resource_stalls) {
        return true;
    }

    if (res == CacheResourceType_TagArray) {
        if (tagArray.tryAccess(addressToCacheSet(addr))) return true;
        else {
            DPRINTF(RubyResourceStalls,
                    "Tag array stall on addr %#x in set %d\n",
                    addr, addressToCacheSet(addr));
            cacheMemoryStats.numTagArrayStalls++;
            return false;
        }
    } else if (res == CacheResourceType_DataArray) {
        if (dataArray.tryAccess(addressToCacheSet(addr))) return true;
        else {
            DPRINTF(RubyResourceStalls,
                    "Data array stall on addr %#x in set %d\n",
                    addr, addressToCacheSet(addr));
            cacheMemoryStats.numDataArrayStalls++;
            return false;
        }
    } else if (res == CacheResourceType_AtomicALUArray) {
        if (atomicALUArray.tryAccess(addr)) return true;
        else {
            DPRINTF(RubyResourceStalls,
                    "Atomic ALU array stall on addr %#x in line address %#x\n",
                    addr, makeLineAddress(addr));
            cacheMemoryStats.numAtomicALUArrayStalls++;
            return false;
        }
    } else {
        panic("Unrecognized cache resource type.");
    }
}

bool
CacheMemory::isBlockInvalid(int64_t cache_set, int64_t loc)
{
  return (m_cache[cache_set][loc]->m_Permission == AccessPermission_Invalid);
}

bool
CacheMemory::isBlockNotBusy(int64_t cache_set, int64_t loc)
{
  return (m_cache[cache_set][loc]->m_Permission != AccessPermission_Busy);
}

/* hardware transactional memory */

void
CacheMemory::htmAbortTransaction()
{
    uint64_t htmReadSetSize = 0;
    uint64_t htmWriteSetSize = 0;

    // iterate through every set and way to get a cache line
    for (auto i = m_cache.begin(); i != m_cache.end(); ++i)
    {
        std::vector<AbstractCacheEntry*> set = *i;

        for (auto j = set.begin(); j != set.end(); ++j)
        {
            AbstractCacheEntry *line = *j;

            if (line != nullptr) {
                htmReadSetSize += (line->getInHtmReadSet() ? 1 : 0);
                htmWriteSetSize += (line->getInHtmWriteSet() ? 1 : 0);
                if (line->getInHtmWriteSet()) {
                    line->invalidateEntry();
                }
                line->setInHtmWriteSet(false);
                line->setInHtmReadSet(false);
                line->clearLocked();
            }
        }
    }

    cacheMemoryStats.htmTransAbortReadSet.sample(htmReadSetSize);
    cacheMemoryStats.htmTransAbortWriteSet.sample(htmWriteSetSize);
    DPRINTF(HtmMem, "htmAbortTransaction: read set=%u write set=%u\n",
        htmReadSetSize, htmWriteSetSize);
}

void
CacheMemory::htmCommitTransaction()
{
    uint64_t htmReadSetSize = 0;
    uint64_t htmWriteSetSize = 0;

    // iterate through every set and way to get a cache line
    for (auto i = m_cache.begin(); i != m_cache.end(); ++i)
    {
        std::vector<AbstractCacheEntry*> set = *i;

        for (auto j = set.begin(); j != set.end(); ++j)
        {
            AbstractCacheEntry *line = *j;
            if (line != nullptr) {
                htmReadSetSize += (line->getInHtmReadSet() ? 1 : 0);
                htmWriteSetSize += (line->getInHtmWriteSet() ? 1 : 0);
                line->setInHtmWriteSet(false);
                line->setInHtmReadSet(false);
                line->clearLocked();
             }
        }
    }

    cacheMemoryStats.htmTransCommitReadSet.sample(htmReadSetSize);
    cacheMemoryStats.htmTransCommitWriteSet.sample(htmWriteSetSize);
    DPRINTF(HtmMem, "htmCommitTransaction: read set=%u write set=%u\n",
        htmReadSetSize, htmWriteSetSize);
}

void
CacheMemory::profileDemandHit()
{
    cacheMemoryStats.m_demand_hits++;
}
void
CacheMemory::profileDemandMiss()
{
    cacheMemoryStats.m_demand_misses++;
}

void
CacheMemory::profilePrefetchHit()
{
    cacheMemoryStats.m_prefetch_hits++;
}

void
CacheMemory::profilePrefetchMiss()
{
    cacheMemoryStats.m_prefetch_misses++;
}



void
CacheMemory::injectBitFlip(int set, int way, int byte, int bit)
{
    // 1. Directly index the 2D array using [set][way]
    AbstractCacheEntry* entry = m_cache[set][way];

    if (entry != nullptr) {
        // 2. Grab the 64-byte data block
        DataBlock& data = entry->getDataBlk();

        // 3. Read the specific byte
        uint8_t val = data.getByte(byte);

        // 4. Flip the exact bit using the XOR (^=) operator
        val ^= (1 << bit);

        // 5. Write the corrupted byte back into the cache line
        data.setByte(byte, val);

        inform("TCP Cache Fault Executed! Set:%d Way:%d Byte:%d Bit:%d (New Byte Val: 0x%x)",
               set, way, byte, bit, val);
    } else {
        warn("TCP Cache Fault Failed: Cache entry was NULL at Set:%d Way:%d", set, way);
    }
}


void
CacheMemory::loadSqcHardFault(int sqc, int set, int way, int byte, int bit,
                              int stuck_val)
{
    SqcHardFault f;
    f.sqc = sqc;
    f.set = set;
    f.way = way;
    f.byte = byte;
    f.bit = bit;
    f.stuck_val = stuck_val;

    sqcHardFaults.push_back(f);

    inform("CacheMemory: LOADED HARD SQC FAULT (SQC:%d Set:%d Way:%d Byte:%d Bit:%d Stuck:%d)",
           sqc, set, way, byte, bit, stuck_val);
}

bool
CacheMemory::enforceSqcFaultOnFetch(
    Addr address,
    DataBlock& data,
    SqcHardFaultHit *hit)
{
    if (hit != nullptr) {
        *hit = SqcHardFaultHit{};
    }

    if (name().find("sqc_cntrl") == std::string::npos) {
        return false;
    }

    Addr line_addr = ruby::makeLineAddress(address, m_start_index_bit);

    bool faulted = false;

    // --------------------------------------------------
    // 1. Active transient SQC fault: SRAM was already flipped
    //    earlier by activateSqcTransientFaultIfMatch().
    //    We only need to return true so Sequencer poisons
    //    the packet and RubyPort preserves the payload.
    // --------------------------------------------------
    if (sqcTransientFault.valid &&
        sqcTransientFault.active &&
        sqcTransientFault.line_addr == line_addr) {
        return true;
    }

    // --------------------------------------------------
    // 2. Hard SQC faults: enforce stuck-at on returned data
    // --------------------------------------------------
    for (const auto& f : sqcHardFaults) {
        if (f.set < 0 || f.set >= (int)m_cache.size()) {
            continue;
        }
        if (f.way < 0 || f.way >= (int)m_cache[f.set].size()) {
            continue;
        }
        if (f.byte < 0 || f.byte >= 64 || f.bit < 0 || f.bit >= 8) {
            continue;
        }

        AbstractCacheEntry* entry = m_cache[f.set][f.way];
        if (entry == nullptr) {
            continue;
        }
        if (entry->m_Permission == AccessPermission_Invalid) {
            continue;
        }

        Addr entry_line_addr =
            ruby::makeLineAddress(entry->m_Address, m_start_index_bit);

        if (entry_line_addr != line_addr) {
            continue;
        }

        uint8_t old_byte = data.getByte(f.byte);
        uint8_t new_byte = old_byte;

        if (f.stuck_val == 1) {
            new_byte |= (1 << f.bit);
        } else {
            new_byte &= ~(1 << f.bit);
        }

        if (hit != nullptr) {
            hit->matched = true;
            hit->sqc = f.sqc;
            hit->set = f.set;
            hit->way = f.way;
            hit->byte = f.byte;
            hit->bit = f.bit;
            hit->stuck_val = f.stuck_val;
            hit->observed_byte = old_byte;
            hit->forced_byte = new_byte;
            hit->forced_now = new_byte != old_byte;
        }

        if (new_byte != old_byte) {
            data.setByte(f.byte, new_byte);
        }

        // Even if byte already had the stuck value, still poison it
        // so RubyPort does not clean it.
        faulted = true;
    }

    return faulted;
}

void
CacheMemory::loadSqcTransientFault(int sqc, int set, int way, int byte, int bit)
{
    sqcTransientFault.valid = true;
    sqcTransientFault.active = false;
    sqcTransientFault.set = set;
    sqcTransientFault.way = way;
    sqcTransientFault.byte = byte;
    sqcTransientFault.bit = bit;
    sqcTransientFault.line_addr = 0;

    inform("CacheMemory: LOADED TRANSIENT SQC FAULT (SQC:%d Set:%d Way:%d Byte:%d Bit:%d)",
           sqc, set, way, byte, bit);
}

bool
CacheMemory::activateSqcTransientFaultIfMatch(int sqc, int set, int way)
{
    inform("CacheMemory: activateSqcTransientFaultIfMatch called SQC=%d call=(Set:%d Way:%d) target=(Set:%d Way:%d) valid=%d active=%d",
           sqc, set, way,
           sqcTransientFault.set, sqcTransientFault.way,
           sqcTransientFault.valid, sqcTransientFault.active);

    if (!(sqcTransientFault.valid &&
          sqcTransientFault.set == set &&
          sqcTransientFault.way == way)) {
        return false;
    }

    if (set < 0 || set >= (int)m_cache.size()) {
        return false;
    }
    if (way < 0 || way >= (int)m_cache[set].size()) {
        return false;
    }

    AbstractCacheEntry* entry = m_cache[set][way];

    if (entry == nullptr) {
        inform("CacheMemory: SQC transient target slot has no entry");
        return false;
    }

    if (entry->m_Permission == AccessPermission_Invalid) {
        inform("CacheMemory: SQC transient target slot entry is invalid");
        return false;
    }

    DataBlock& blk = entry->getDataBlk();
    uint8_t old_val = blk.getByte(sqcTransientFault.byte);
    uint8_t new_val = old_val ^ (1 << sqcTransientFault.bit);

    blk.setByte(sqcTransientFault.byte, new_val);
    sqcTransientFault.active = true;
    sqcTransientFault.line_addr =
        ruby::makeLineAddress(entry->m_Address, m_start_index_bit);

    inform("SQC TRANSIENT EVENT FIRED! Set:%d Way:%d LineAddr:%#lx Byte:%d Bit:%d (0x%x -> 0x%x)",
           set, way, sqcTransientFault.line_addr,
           sqcTransientFault.byte, sqcTransientFault.bit,
           old_val, new_val);

    return true;
}


bool
CacheMemory::sqcTransientFaultAffectsFetch(Addr address) const
{
    if (!(sqcTransientFault.valid && sqcTransientFault.active)) {
        return false;
    }

    Addr line_addr = ruby::makeLineAddress(address, m_start_index_bit);

    if (line_addr == sqcTransientFault.line_addr) {
        inform("CacheMemory: Fetch from Active Transient SQC line! Addr:%#lx",
               address);
        return true;
    }

    return false;
}

void
CacheMemory::clearSqcTransientFaultIfMatch(int set, int way, Addr line_addr)
{
    if (!(sqcTransientFault.valid &&
          sqcTransientFault.active &&
          sqcTransientFault.set == set &&
          sqcTransientFault.way == way)) {
        return;
    }

    Addr normalized = (line_addr == 0) ? 0 :
        ruby::makeLineAddress(line_addr, m_start_index_bit);

    if (normalized == 0 || sqcTransientFault.line_addr == normalized) {
        inform("CacheMemory: CLEARING Transient SQC Fault! Set:%d Way:%d LineAddr:%#lx",
               set, way, sqcTransientFault.line_addr);
        clearSqcTransientFault();
    }
}

void
CacheMemory::clearSqcTransientFault()
{
    sqcTransientFault.valid = false;
    sqcTransientFault.active = false;
    sqcTransientFault.set = -1;
    sqcTransientFault.way = -1;
    sqcTransientFault.byte = -1;
    sqcTransientFault.bit = -1;
    sqcTransientFault.line_addr = 0;
}

void CacheMemory::loadTcpHardFault(int set, int way, int byte, int bit, int stuck_val) {
    tcpHardFaults.push_back({set, way, byte, bit, stuck_val});
    inform("CacheMemory: SUCCESSFULLY LOADED HARD TCP FAULT (Set:%d Way:%d Byte:%d Bit:%d Stuck:%d)",
           set, way, byte, bit, stuck_val);
}


void
CacheMemory::applySqcHardFaults()
{


    for (auto& f : sqcHardFaults) {

        // 1. Bounds check for set/way
        if (f.set < 0 || f.set >= (int)m_cache.size()) {
            continue;
        }
        if (f.way < 0 || f.way >= (int)m_cache[f.set].size()) {
            continue;
        }

        AbstractCacheEntry* entry = m_cache[f.set][f.way];

        // 2. Entry must exist and be valid
        if (entry == nullptr ||
            entry->m_Permission == AccessPermission_Invalid) {
            continue;
        }

        // 3. Bounds check for byte/bit
        if (f.byte < 0 || f.byte >= 64 || f.bit < 0 || f.bit >= 8) {
            continue;
        }

        DataBlock& blk = entry->getDataBlk();
        uint8_t old_byte = blk.getByte(f.byte);
        uint8_t new_byte = old_byte;

        // 4. Enforce stuck-at logic
        if (f.stuck_val == 1) {
            new_byte |= (1 << f.bit);
        } else {
            new_byte &= ~(1 << f.bit);
        }

        // 5. Only write/log if something changed
        if (new_byte != old_byte) {
            blk.setByte(f.byte, new_byte);

            if (Shader::propagationVerboseEnabledGlobal()) {
                inform("LIVE SQC HARD FAULT HIT! Addr: %#lx | Set:%d Way:%d Byte:%d Bit:%d | Old: 0x%x -> New: 0x%x",
                       entry->m_Address, f.set, f.way, f.byte, f.bit,
                       old_byte, new_byte);
            }
        }
    }
}


void CacheMemory::applyTcpHardFaults() {
    // This perfectly mimics physical broken wires, but with strict validation
    for (auto& f : tcpHardFaults) {

        // 1. BOUNDS CHECK: Ensure Set and Way exist in the physical array
        if (f.set < m_cache.size() && f.way < m_cache[f.set].size()) {
            AbstractCacheEntry* entry = m_cache[f.set][f.way];

            // 2. LIVENESS CHECK: Slot must exist AND not be marked as Invalid
            if (entry != nullptr && entry->m_Permission != AccessPermission_Invalid) {

                // 3. BOUNDS CHECK: Ensure Byte and Bit are valid for a 64B cache line
                if (f.byte < 64 && f.bit < 8) {
                    DataBlock& blk = entry->getDataBlk();
                    uint8_t old_byte = blk.getByte(f.byte);
                    uint8_t new_byte = old_byte;

                    // Enforce Stuck-At logic
                    if (f.stuck_val == 1) {
                        new_byte |= (1 << f.bit);  // Force bit to 1
                    } else {
                        new_byte &= ~(1 << f.bit); // Force bit to 0
                    }

                    // 4. OBSERVABILITY CHECK: Only write back and log if it ACTUALLY changed
                    if (new_byte != old_byte) {
                        blk.setByte(f.byte, new_byte);

                      inform("LIVE HARD FAULT HIT! Addr: %#lx | Set:%d Way:%d Byte:%d Bit:%d | Old: 0x%x -> New: 0x%x",
           entry->m_Address, f.set, f.way, f.byte, f.bit, old_byte, new_byte);
                    }
                }
            }
        }
    }
}

bool
CacheMemory::enforceFaultOnLoad(
    Addr address,
    DataBlock& data,
    TcpHardFaultHit *hit)
{
    // Καθαρίζουμε το αποτέλεσμα πριν από κάθε lookup.
    if (hit != nullptr) {
        *hit = TcpHardFaultHit{};
    }

    const int64_t cacheSet = addressToCacheSet(address);
    const int way = findTagInSet(cacheSet, address);

    if (way == -1) {
        return false;
    }

    bool matched_faulted_slot = false;

    for (auto& f : tcpHardFaults) {
        if (f.set != cacheSet || f.way != way) {
            continue;
        }

        matched_faulted_slot = true;

        if (f.byte >= 64 || f.bit >= 8) {
            continue;
        }

        const uint8_t old_byte = data.getByte(f.byte);
        uint8_t new_byte = old_byte;

        if (f.stuck_val == 1) {
            new_byte |= static_cast<uint8_t>(1U << f.bit);
        } else {
            new_byte &= static_cast<uint8_t>(~(1U << f.bit));
        }

        // Επιστρέφουμε τα στοιχεία στο GPUCoalescer.
        if (hit != nullptr) {
            hit->matched = true;
            hit->set = static_cast<int>(cacheSet);
            hit->way = way;
            hit->byte = f.byte;
            hit->bit = f.bit;
            hit->stuck_val = f.stuck_val;
            hit->observed_byte = old_byte;
            hit->forced_byte = new_byte;
            hit->forced_now = (old_byte != new_byte);
        }

        // Κρατάμε το returned payload forced.
        if (old_byte != new_byte) {
            data.setByte(f.byte, new_byte);
        }

        // Κρατάμε forced και το backing SRAM.
        auto *entry = m_cache[cacheSet][way];

        if (entry != nullptr &&
            entry->m_Permission != AccessPermission_Invalid) {

            const uint8_t sram_old =
                entry->getDataBlk().getByte(f.byte);

            uint8_t sram_new = sram_old;

            if (f.stuck_val == 1) {
                sram_new |= static_cast<uint8_t>(1U << f.bit);
            } else {
                sram_new &=
                    static_cast<uint8_t>(~(1U << f.bit));
            }

            if (sram_old != sram_new) {
                entry->getDataBlk().setByte(f.byte, sram_new);
            }
        }

        // Κρατάμε το παλιό μήνυμα για compatibility.
        if (Shader::propagationVerboseEnabledGlobal()) {
        inform(
            "SLICC INTERCEPT: Hard Fault response from faulted slot! "
            "Addr: %#lx | Set:%lld Way:%d Byte:%d Bit:%d",
            address,
            cacheSet,
            way,
            f.byte,
            f.bit);

        // Structured per-access records are debug-only.
        inform(
            "FAULT_TRACE event=TCP_SLOT_HIT fault_id=0 "
            "tick=%llu line=0x%llx set=%lld way=%d "
            "byte=%d bit=%d stuck=%d "
            "observed_byte=0x%02x forced_byte=0x%02x "
            "forced_now=%d",
            static_cast<unsigned long long>(curTick()),
            static_cast<unsigned long long>(address),
            cacheSet,
            way,
            f.byte,
            f.bit,
            f.stuck_val,
            static_cast<unsigned int>(old_byte),
            static_cast<unsigned int>(new_byte),
            old_byte != new_byte ? 1 : 0);
        }
    }

    return matched_faulted_slot;
}

// ========================================================================
// TRANSIENT FAULT IMPLEMENTATION
// ========================================================================

void
CacheMemory::loadTcpTransientFault(int set, int way, int byte, int bit)
{
    tcpTransientFault.valid = true;
    tcpTransientFault.active = false; // Not active until the scheduled event fires
    tcpTransientFault.set = set;
    tcpTransientFault.way = way;
    tcpTransientFault.byte = byte;
    tcpTransientFault.bit = bit;
    tcpTransientFault.line_addr = 0;

    inform("CacheMemory: LOADED TRANSIENT TCP FAULT (Set:%d Way:%d Byte:%d Bit:%d)",
           set, way, byte, bit);
}

bool
CacheMemory::activateTcpTransientFaultIfMatch(int set, int way)
{

inform("CacheMemory: activateTcpTransientFaultIfMatch called call=(Set:%d Way:%d) target=(Set:%d Way:%d) valid=%d active=%d",
       set, way,
       tcpTransientFault.set, tcpTransientFault.way,
       tcpTransientFault.valid, tcpTransientFault.active);

    if (!(tcpTransientFault.valid &&
          tcpTransientFault.set == set &&
          tcpTransientFault.way == way)) {
        return false;
    }

    AbstractCacheEntry* entry = m_cache[set][way];

    if (entry == nullptr || entry->m_Permission == AccessPermission_Invalid) {
        warn("Transient Event Fired, but cache slot Set:%d Way:%d is EMPTY/INVALID. Fault masked.",
             set, way);
        clearTcpTransientFault();
        return false;
    }

    if (tcpTransientFault.byte < 0 || tcpTransientFault.byte >= 64 ||
        tcpTransientFault.bit  < 0 || tcpTransientFault.bit  >= 8) {
        warn("Transient TCP fault has invalid Byte/Bit (%d,%d)",
             tcpTransientFault.byte, tcpTransientFault.bit);
        clearTcpTransientFault();
        return false;
    }

    tcpTransientFault.line_addr = makeLineAddress(entry->m_Address);
    tcpTransientFault.active = true;

    DataBlock& data = entry->getDataBlk();
    uint8_t old_val = data.getByte(tcpTransientFault.byte);
    uint8_t new_val = old_val ^ (1 << tcpTransientFault.bit);
    data.setByte(tcpTransientFault.byte, new_val);

    inform("TRANSIENT EVENT FIRED! Set:%d Way:%d LineAddr:%#lx Byte:%d Bit:%d (0x%x -> 0x%x)",
           set, way, tcpTransientFault.line_addr,
           tcpTransientFault.byte, tcpTransientFault.bit,
           old_val, new_val);

    return true;
}

bool
CacheMemory::transientFaultAffectsLoad(Addr address) const
{
    if (!tcpTransientFault.active) {
        return false;
    }

    Addr line_addr = makeLineAddress(address);

    if (line_addr == tcpTransientFault.line_addr) {
        inform("SLICC INTERCEPT: Read from Active Transient Fault slot! Addr:%#lx",
               address);
        return true;
    }

    return false;
}

void
CacheMemory::clearTcpTransientFaultIfMatch(int set, int way, Addr line_addr)
{
    if (!(tcpTransientFault.active &&
          tcpTransientFault.set == set &&
          tcpTransientFault.way == way)) {
        return;
    }

    Addr normalized = (line_addr == 0) ? 0 : makeLineAddress(line_addr);

    if (tcpTransientFault.line_addr == normalized || normalized == 0) {
        inform("CacheMemory: CLEARING Transient Fault! Set:%d Way:%d LineAddr:%#lx was overwritten or evicted.",
               set, way, tcpTransientFault.line_addr);
        clearTcpTransientFault();
    }
}

void
CacheMemory::clearTcpTransientFault()
{
    tcpTransientFault.valid = false;
    tcpTransientFault.active = false;
    tcpTransientFault.set = -1;
    tcpTransientFault.way = -1;
    tcpTransientFault.byte = -1;
    tcpTransientFault.bit = -1;
    tcpTransientFault.line_addr = 0;
}

void
CacheMemory::loadTccHardFault(int set, int way, int byte, int bit, int stuckValue)
{
    TccHardFault f;
    f.set = set;
    f.way = way;
    f.byte = byte;
    f.bit = bit;
    f.stuckValue = stuckValue;
    tccHardFaults.push_back(f);


    enforceTccHardFaultsOnLine(set, way);

    inform("Loaded HARD TCC fault: set=%d way=%d byte=%d bit=%d stuck=%d",
           set, way, byte, bit, stuckValue);
}


bool
CacheMemory::enforceTccHardFaultsOnLine(int set, int way)
{
    // --- OPTIMIZATION 1: Fast-Path Exit ---
    // If no faults are loaded, instantly exit to save billions of CPU cycles
    if (tccHardFaults.empty()) return false;

    if (set < 0 || set >= m_cache_num_sets) return false;
    if (way < 0 || way >= m_cache_assoc) return false;

    AbstractCacheEntry *entry = m_cache[set][way];

    if (entry == nullptr || entry->m_Permission == AccessPermission_Invalid) {
        return false;
    }

    bool fault_applied = false;
    DataBlock &blk = entry->getDataBlk();

    for (const auto &f : tccHardFaults) {
        if (f.set != set || f.way != way) {
            continue;
        }

        uint8_t old_byte = blk.getByte(f.byte);
        uint8_t mask = static_cast<uint8_t>(1u << f.bit);
        uint8_t new_byte = f.stuckValue ? (old_byte | mask)
                                        : (old_byte & ~mask);

       if (new_byte != old_byte) {
            blk.setByte(f.byte, new_byte);
        }

        fault_applied = true;

        // --- THE POISON BRIDGE ANNOUNCEMENT ---
        Addr line_addr = makeLineAddress(entry->m_Address);

        // Only print ONCE when it first becomes poisoned to avoid terminal spam
        if (!CacheMemory::isTccPoisoned(line_addr)) {
            CacheMemory::addTccPoison(line_addr);
            inform("TCC POISON BRIDGE: Physically corrupted Line %#lx in L2 SRAM and added to whiteboard!", line_addr);
        }
    }

    return fault_applied;
}

void
CacheMemory::loadTccTransientFault(int set, int way, int byte, int bit)
{
    tccTransientFault.valid = true;
    tccTransientFault.active = false;
    tccTransientFault.set = set;
    tccTransientFault.way = way;
    tccTransientFault.byte = byte;
    tccTransientFault.bit = bit;
    tccTransientFault.line_addr = 0;

    inform("CacheMemory: LOADED TRANSIENT TCC FAULT Set=%d Way=%d Byte=%d Bit=%d",
           set, way, byte, bit);
}

bool
CacheMemory::activateTccTransientFaultIfMatch(int set, int way)
{
    // --- NEW: PREVENT DOUBLE-FLIPPING! ---
    // If it already flipped the bit, exit immediately so it doesn't heal it.
    if (tccTransientFault.active) {
        return false;
    }
    // -------------------------------------

    inform("CacheMemory: activateTccTransientFaultIfMatch call=(Set:%d Way:%d) target=(Set:%d Way:%d) valid=%d active=%d",
           set, way,
           tccTransientFault.set, tccTransientFault.way,
           tccTransientFault.valid, tccTransientFault.active);

    if (!(tccTransientFault.valid &&
          tccTransientFault.set == set &&
          tccTransientFault.way == way)) {
        return false;
    }

    if (set < 0 || set >= m_cache_num_sets) return false;
    if (way < 0 || way >= m_cache_assoc) return false;

    AbstractCacheEntry *entry = m_cache[set][way];

    if (entry == nullptr || entry->m_Permission == AccessPermission_Invalid) {
        warn("Transient TCC event fired, but Set:%d Way:%d is empty/invalid.",
             set, way);
        clearTccTransientFault();
        return false;
    }

    if (tccTransientFault.byte < 0 ||
        tccTransientFault.byte >= m_block_size ||
        tccTransientFault.bit < 0 ||
        tccTransientFault.bit >= 8) {
        warn("Invalid transient TCC Byte/Bit (%d,%d)",
             tccTransientFault.byte, tccTransientFault.bit);
        clearTccTransientFault();
        return false;
    }

    DataBlock &blk = entry->getDataBlk();

    uint8_t old_val = blk.getByte(tccTransientFault.byte);
    uint8_t new_val = old_val ^ static_cast<uint8_t>(1u << tccTransientFault.bit);

    blk.setByte(tccTransientFault.byte, new_val);

    tccTransientFault.line_addr = makeLineAddress(entry->m_Address);
    tccTransientFault.active = true;

    CacheMemory::addTccPoison(tccTransientFault.line_addr);

    inform("[TCC-TRANSIENT-FLIP] line=%#lx set=%d way=%d byte=%d bit=%d old=%#x new=%#x",
           tccTransientFault.line_addr,
           set, way,
           tccTransientFault.byte,
           tccTransientFault.bit,
           old_val, new_val);

    return true;
}
bool
CacheMemory::tccTransientFaultAffectsLoad(Addr address) const
{
    if (!(tccTransientFault.valid && tccTransientFault.active)) {
        return false;
    }

    Addr line_addr = makeLineAddress(address);

    if (line_addr == tccTransientFault.line_addr) {
        inform("[TCC-TRANSIENT-LOAD-HIT] addr=%#lx line=%#lx",
               address, line_addr);
        return true;
    }

    return false;
}


bool
CacheMemory::tccTransientFaultAffectsFetch(Addr address) const
{
    return tccTransientFaultAffectsLoad(address);
}


void
CacheMemory::clearTccTransientFaultIfMatch(int set, int way, Addr line_addr)
{
    if (!(tccTransientFault.valid &&
          tccTransientFault.active &&
          tccTransientFault.set == set &&
          tccTransientFault.way == way)) {
        return;
    }

    Addr normalized = (line_addr == 0) ? 0 : makeLineAddress(line_addr);

    if (normalized == 0 || normalized == tccTransientFault.line_addr) {
        inform("CacheMemory: CLEARING transient TCC fault Set=%d Way=%d Line=%#lx",
               set, way, tccTransientFault.line_addr);

        CacheMemory::removeTccPoison(tccTransientFault.line_addr);
        clearTccTransientFault();
    }
}


void
CacheMemory::clearTccTransientFault()
{
    tccTransientFault.valid = false;
    tccTransientFault.active = false;
    tccTransientFault.set = -1;
    tccTransientFault.way = -1;
    tccTransientFault.byte = -1;
    tccTransientFault.bit = -1;
    tccTransientFault.line_addr = 0;
}


} // namespace ruby
} // namespace gem5
