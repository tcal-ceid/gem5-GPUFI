/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __GPU_COMPUTE_VRF_PROPAGATION_HH__
#define __GPU_COMPUTE_VRF_PROPAGATION_HH__

#include <cstdint>

namespace gem5
{

using VrfPropagationEventId = uint64_t;

static constexpr VrfPropagationEventId NoVrfPropagationEvent = 0;

enum class RegisterPropagationOrigin : uint8_t
{
    Unknown,
    Vrf,
    Srf,
    Mixed
};

inline const char *
registerPropagationOriginName(RegisterPropagationOrigin origin)
{
    switch (origin) {
      case RegisterPropagationOrigin::Vrf:
        return "VRF";
      case RegisterPropagationOrigin::Srf:
        return "SRF";
      case RegisterPropagationOrigin::Mixed:
        return "MIXED";
      case RegisterPropagationOrigin::Unknown:
      default:
        return "UNKNOWN";
    }
}

/**
 * Provenance metadata associated with one physical VGPR lane.
 *
 * depth counts destination-producing instructions:
 *   source activation = 0
 *   first descendant register write = 1
 *   next descendant register write = 2
 */
struct VrfPropagationTag
{
    VrfPropagationEventId eventId = NoVrfPropagationEvent;
    VrfPropagationEventId rootActivationId =
        NoVrfPropagationEvent;

    /*
     * True when several different effective source activations
     * have merged into this value. In that case rootActivationId
     * remains zero instead of selecting one root arbitrarily.
     */
    bool multipleActivationRoots = false;

    RegisterPropagationOrigin originComponent =
        RegisterPropagationOrigin::Unknown;

    uint32_t depth = 0;

    int originCu = -1;
    int originSimd = -1;
    int originWfDynId = -1;
    int originLane = -1;

    /*
     * Dynamic location currently owning this descendant value.
     * This prevents stale metadata when physical VGPRs are reused.
     */
    int currentCu = -1;
    int currentSimd = -1;
    int currentWfDynId = -1;
    int currentLane = -1;

    bool
    valid() const
    {
        return eventId != NoVrfPropagationEvent;
    }
};

enum class VrfPropagationScope
{
    SameLane,
    CrossLaneSameWavefront,
    CrossWavefrontSameCu,
    CrossCu
};

} // namespace gem5

#endif // __GPU_COMPUTE_VRF_PROPAGATION_HH__
