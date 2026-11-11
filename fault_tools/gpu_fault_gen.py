import argparse
import json
import random
from typing import (
    Any,
    Dict,
    List,
    Optional,
)


def clamp_int(x: float, lo: int, hi: int) -> int:
    """Rounds a float to the nearest integer and clamps it to [lo, hi]."""
    xi = int(round(x))
    return max(lo, min(hi, xi))


def load_config(path: str) -> Dict[str, Any]:
    """Loads the GPU microarchitectural geometry configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


def gaussian_tick(
    rng: random.Random,
    mu: float,
    sigma: float,
    tick_start: int,
    tick_end: int,
) -> int:
    """Samples an injection tick from a Gaussian distribution clamped to [tick_start, tick_end - 1]."""
    t = rng.gauss(mu, sigma)
    return clamp_int(t, tick_start, tick_end - 1)


def _is_vrf_component(comp: Dict[str, Any]) -> bool:
    # VRF needs lanes_per_reg and bits_per_lane
    return "lanes_per_reg" in comp and "bits_per_lane" in comp


def _is_srf_component(comp: Dict[str, Any]) -> bool:
    # SRF is scalar: bits_per_reg (typically 32) and no lane fields
    return "bits_per_reg" in comp and not _is_vrf_component(comp)


def _is_tcp_cache_component(comp: Dict[str, Any]) -> bool:
    # TCP has a "cu" instance
    return (
        "sets" in comp
        and "assoc" in comp
        and "cu" in comp.get("instances", {})
    )


def _is_sqc_cache_component(comp: Dict[str, Any]) -> bool:
    # SQC has an "sqc" instance
    return (
        "sets" in comp
        and "assoc" in comp
        and "sqc" in comp.get("instances", {})
    )
    # ----------------------------


def _is_tcc_cache_component(comp: Dict[str, Any]) -> bool:
    # TCC has a "tcc" instance
    return (
        "sets" in comp
        and "assoc" in comp
        and "tcc" in comp.get("instances", {})
    )


# L1 Cache (TCP) fault generation
# ----------------------------
def gen_transient_cache_fault(
    rng: random.Random, comp: Dict[str, Any], tick: int
) -> Dict[str, Any]:
    """Generates a transient single-bit fault descriptor for the L1 Vector Data Cache (TCP)."""
    num_cu = comp["instances"]["cu"]
    num_sets = comp["sets"]
    assoc = comp["assoc"]
    line_size = comp.get("line_size", 64)

    cu = rng.randrange(num_cu)
    set_idx = rng.randrange(num_sets)
    way_idx = rng.randrange(assoc)
    byte_offset = rng.randrange(line_size)
    bit_in_byte = rng.randrange(8)

    return {
        "tick": tick,
        "location": {
            "type": "cache",
            "cache_type": "tcp",
            "cu": cu,
            "set": set_idx,
            "way": way_idx,
            "byte": byte_offset,
            "bit": bit_in_byte,
        },
    }


def gen_hard_cache_fault(
    rng: random.Random, comp: Dict[str, Any], stuck_value: Optional[int]
) -> Dict[str, Any]:
    """Generates a permanent stuck-at fault descriptor for the L1 Vector Data Cache (TCP)."""
    num_cu = comp["instances"]["cu"]
    num_sets = comp["sets"]
    assoc = comp["assoc"]
    line_size = comp.get("line_size", 64)

    cu = rng.randrange(num_cu)
    set_idx = rng.randrange(num_sets)
    way_idx = rng.randrange(assoc)
    byte_offset = rng.randrange(line_size)
    bit_in_byte = rng.randrange(8)

    sv = stuck_value if stuck_value is not None else rng.randrange(2)

    return {
        "location": {
            "type": "cache",
            "cache_type": "tcp",
            "cu": cu,
            "set": set_idx,
            "way": way_idx,
            "byte": byte_offset,
            "bit": bit_in_byte,
        },
        "mode": "stuck_at",
        "stuck_value": sv,
    }


# ----------------------------
# VRF fault generation
# ----------------------------
def gen_transient_vrf_fault(
    rng: random.Random, comp: Dict[str, Any], tick: int
) -> Dict[str, Any]:
    """Generates a transient single-bit fault descriptor for the Vector Register File (VRF)."""
    num_cu = comp["instances"]["cu"]
    num_simd = comp["instances"]["simd"]
    num_regs = comp["num_regs"]
    lanes = comp["lanes_per_reg"]
    bits_per_lane = comp["bits_per_lane"]

    cu = rng.randrange(num_cu)
    simd = rng.randrange(num_simd)
    phys_reg_index = rng.randrange(num_regs)
    lane = rng.randrange(lanes)
    lane_bit = rng.randrange(bits_per_lane)
    bit_index = lane * bits_per_lane + lane_bit

    return {
        "tick": tick,
        "location": {
            "type": "regfile",
            "rf": "vrf",
            "instance": {"cu": cu, "simd": simd},
            "phys_reg_index": phys_reg_index,
            "lane": lane,
            "lane_bit": lane_bit,
            "bit_index": bit_index,
        },
    }


def gen_hard_vrf_fault(
    rng: random.Random, comp: Dict[str, Any], stuck_value: Optional[int]
) -> Dict[str, Any]:
    """Generates a permanent stuck-at fault descriptor for the Vector Register File (VRF)."""
    num_cu = comp["instances"]["cu"]
    num_simd = comp["instances"]["simd"]
    num_regs = comp["num_regs"]
    lanes = comp["lanes_per_reg"]
    bits_per_lane = comp["bits_per_lane"]

    cu = rng.randrange(num_cu)
    simd = rng.randrange(num_simd)
    phys_reg_index = rng.randrange(num_regs)
    lane = rng.randrange(lanes)
    lane_bit = rng.randrange(bits_per_lane)
    bit_index = lane * bits_per_lane + lane_bit

    sv = stuck_value if stuck_value is not None else rng.randrange(2)

    return {
        "location": {
            "type": "regfile",
            "rf": "vrf",
            "instance": {"cu": cu, "simd": simd},
            "phys_reg_index": phys_reg_index,
            "lane": lane,
            "lane_bit": lane_bit,
            "bit_index": bit_index,
        },
        "mode": "stuck_at",
        "stuck_value": sv,
    }


# ----------------------------
# SRF fault generation
# ----------------------------
def gen_transient_srf_fault(
    rng: random.Random, comp: Dict[str, Any], tick: int
) -> Dict[str, Any]:
    """Generates a transient single-bit fault descriptor for the Scalar Register File (SRF)."""
    num_cu = comp["instances"]["cu"]
    num_simd = comp["instances"]["simd"]
    num_regs = comp["num_regs"]
    bits_per_reg = comp.get("bits_per_reg", 32)

    if bits_per_reg <= 0:
        raise ValueError("bits_per_reg must be > 0 for SRF component")

    cu = rng.randrange(num_cu)
    simd = rng.randrange(num_simd)
    phys_reg_index = rng.randrange(num_regs)
    bit = rng.randrange(bits_per_reg)

    return {
        "tick": tick,
        "location": {
            "type": "regfile",
            "rf": "srf",
            "instance": {"cu": cu, "simd": simd},
            "phys_reg_index": phys_reg_index,
            "bit_index": bit,
        },
    }


def gen_hard_srf_fault(
    rng: random.Random, comp: Dict[str, Any], stuck_value: Optional[int]
) -> Dict[str, Any]:
    """Generates a permanent stuck-at fault descriptor for the Scalar Register File (SRF)."""
    num_cu = comp["instances"]["cu"]
    num_simd = comp["instances"]["simd"]
    num_regs = comp["num_regs"]
    bits_per_reg = comp.get("bits_per_reg", 32)

    if bits_per_reg <= 0:
        raise ValueError("bits_per_reg must be > 0 for SRF component")

    cu = rng.randrange(num_cu)
    simd = rng.randrange(num_simd)
    phys_reg_index = rng.randrange(num_regs)
    bit = rng.randrange(bits_per_reg)

    sv = stuck_value if stuck_value is not None else rng.randrange(2)

    return {
        "location": {
            "type": "regfile",
            "rf": "srf",
            "instance": {"cu": cu, "simd": simd},
            "phys_reg_index": phys_reg_index,
            "bit_index": bit,
        },
        "mode": "stuck_at",
        "stuck_value": sv,
    }


def main():
    """CLI entry point: parses campaign arguments and writes deterministic JSON fault masks."""
    ap = argparse.ArgumentParser(
        description="gem5 GPU fault mask generator (transient + hard, VRF, SRF, or Cache)"
    )
    ap.add_argument(
        "--config",
        required=True,
        help="GPU geometry config JSON (e.g., gpu_config_regfiles.json)",
    )
    ap.add_argument(
        "--component",
        default="vgpr_file",
        help="Component name in config (e.g., vgpr_file, sgpr_file, or tcp_cache)",
    )
    ap.add_argument(
        "--fault-model", choices=["transient", "hard"], default="transient"
    )
    ap.add_argument("--num-sims", type=int, required=True)
    ap.add_argument("--faults-per-sim", type=int, required=True)

    ap.add_argument(
        "--tick-start",
        type=int,
        default=None,
        help="Transient only: start of the tick window for Gaussian sampling (inclusive). Defaults to 0.",
    )

    ap.add_argument(
        "--tick-end",
        type=int,
        default=None,
        help="Transient only: end of the tick window for Gaussian sampling (exclusive). Defaults to max_ticks.",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base seed (per-sim seed = seed + sim_id)",
    )
    ap.add_argument(
        "--sigma-frac",
        type=float,
        default=6.0,
        help="Transient only: sigma = max_ticks / sigma_frac (default 6 => sigma=max/6)",
    )
    ap.add_argument(
        "--stuck-value",
        type=int,
        choices=[0, 1],
        default=None,
        help="Hard only: stuck-at value (0/1). If omitted, random per fault.",
    )
    ap.add_argument("--output", required=True, help="Output JSON file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    comps = cfg.get("components", {})
    if args.component not in comps:
        raise SystemExit(
            f"Component '{args.component}' not found in config. Available: {list(comps.keys())}"
        )
    comp = comps[args.component]

    # Decide whether this component is VRF, SRF, or Cache based on config fields
    if _is_vrf_component(comp):
        rf_kind = "vrf"
    elif _is_srf_component(comp):
        rf_kind = "srf"
    elif _is_sqc_cache_component(comp):
        rf_kind = "sqc"
    elif _is_tcc_cache_component(comp):
        rf_kind = "tcc"
    elif _is_tcp_cache_component(comp):
        rf_kind = "cache"
    else:
        raise SystemExit(
            f"Component '{args.component}' does not look like VRF, SRF, SQC,TCC or TCP. "
            f"Check instances and required fields."
        )

    # Initialize Gaussian parameters for transient faults
    mu, sigma, tick_start, tick_end, max_ticks = 0, 0, 0, 0, 0

    if args.fault_model == "transient":
        if args.sigma_frac <= 0:
            raise SystemExit("--sigma-frac must be > 0 for transient faults")

        tick_start = args.tick_start if args.tick_start is not None else 0
        if args.tick_end is None:
            raise SystemExit("--tick-end is required for transient faults")

        tick_end = args.tick_end

        if tick_start < 0:
            raise SystemExit("--tick-start must be >= 0")
        if tick_start >= tick_end:
            raise SystemExit("--tick-start must be < --tick-end")

        window = tick_end - tick_start
        mu = tick_start + (window / 2.0)
        sigma = window / float(args.sigma_frac)

    out: List[Dict[str, Any]] = []

    for sim_id in range(args.num_sims):
        rng = random.Random(args.seed + sim_id)
        faults: List[Dict[str, Any]] = []

        if args.fault_model == "transient":
            for _ in range(args.faults_per_sim):
                t = gaussian_tick(rng, mu, sigma, tick_start, tick_end)
                if rf_kind == "vrf":
                    faults.append(gen_transient_vrf_fault(rng, comp, t))
                elif rf_kind == "srf":
                    faults.append(gen_transient_srf_fault(rng, comp, t))
                elif rf_kind == "sqc":
                    faults.append(gen_transient_sqc_fault(rng, comp, t))
                elif rf_kind == "tcc":
                    faults.append(gen_transient_tcc_fault(rng, comp, t))
                else:  # rf_kind == "cache"
                    faults.append(gen_transient_cache_fault(rng, comp, t))

            faults.sort(key=lambda f: f["tick"])

            mask: Dict[str, Any] = {
                "id": sim_id,
                "fault_model": "transient",
                "component": args.component,
                "rf": rf_kind,
                "num_faults": len(faults),
                "faults": faults,
                "time_unit": "tick",
                "gaussian": {
                    "mu": mu,
                    "sigma": sigma,
                    "tick_start": tick_start,
                    "tick_end": tick_end,
                },
            }
        else:
            for _ in range(args.faults_per_sim):
                if rf_kind == "vrf":
                    faults.append(
                        gen_hard_vrf_fault(rng, comp, args.stuck_value)
                    )
                elif rf_kind == "srf":
                    faults.append(
                        gen_hard_srf_fault(rng, comp, args.stuck_value)
                    )
                elif rf_kind == "sqc":
                    faults.append(
                        gen_hard_sqc_fault(rng, comp, args.stuck_value)
                    )
                elif rf_kind == "tcc":
                    faults.append(
                        gen_hard_tcc_fault(rng, comp, args.stuck_value)
                    )
                else:  # rf_kind == "cache"
                    faults.append(
                        gen_hard_cache_fault(rng, comp, args.stuck_value)
                    )

            mask = {
                "id": sim_id,
                "fault_model": "hard",
                "component": args.component,
                "rf": rf_kind,
                "num_faults": len(faults),
                "faults": faults,
            }

        out.append(mask)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.num_sims} masks to {args.output}")


# ----------------------------
# SQC & TCC Cache fault generation
# ----------------------------
def gen_transient_sqc_fault(
    rng: random.Random, comp: Dict[str, Any], tick: int
) -> Dict[str, Any]:
    """Generates a transient single-bit fault descriptor for the L1 Instruction Cache (SQC)."""
    num_sqc = comp["instances"]["sqc"]
    num_sets = comp["sets"]
    assoc = comp["assoc"]
    line_size = comp.get("line_size", 64)

    sqc = rng.randrange(num_sqc)
    set_idx = rng.randrange(num_sets)
    way_idx = rng.randrange(assoc)
    byte_offset = rng.randrange(line_size)
    bit_in_byte = rng.randrange(8)

    return {
        "tick": tick,
        "location": {
            "type": "cache",
            "cache_type": "sqc",
            "sqc": sqc,
            "set": set_idx,
            "way": way_idx,
            "byte": byte_offset,
            "bit": bit_in_byte,
        },
    }


def gen_transient_tcc_fault(
    rng: random.Random, comp: Dict[str, Any], tick: int
) -> Dict[str, Any]:
    """Generates a transient single-bit fault descriptor for the L2 Cache (TCC)."""
    num_tcc = comp["instances"]["tcc"]
    num_sets = comp["sets"]
    assoc = comp["assoc"]
    line_size = comp.get("line_size", 64)

    tcc = rng.randrange(num_tcc)
    set_idx = rng.randrange(num_sets)
    way_idx = rng.randrange(assoc)
    byte_offset = rng.randrange(line_size)
    bit_in_byte = rng.randrange(8)

    return {
        "tick": tick,
        "location": {
            "type": "cache",
            "cache_type": "tcc",
            "tcc": tcc,
            "set": set_idx,
            "way": way_idx,
            "byte": byte_offset,
            "bit": bit_in_byte,
        },
    }


def gen_hard_tcc_fault(
    rng: random.Random, comp: Dict[str, Any], stuck_value: Optional[int]
) -> Dict[str, Any]:
    """Generates a permanent stuck-at fault descriptor for the L2 Cache (TCC)."""
    num_tcc = comp["instances"]["tcc"]
    num_sets = comp["sets"]
    assoc = comp["assoc"]
    line_size = comp.get("line_size", 64)

    tcc = rng.randrange(num_tcc)
    set_idx = rng.randrange(num_sets)
    way_idx = rng.randrange(assoc)
    byte_offset = rng.randrange(line_size)
    bit_in_byte = rng.randrange(8)

    sv = stuck_value if stuck_value is not None else rng.randrange(2)

    return {
        "location": {
            "type": "cache",
            "cache_type": "tcc",
            "tcc": tcc,
            "set": set_idx,
            "way": way_idx,
            "byte": byte_offset,
            "bit": bit_in_byte,
        },
        "mode": "stuck_at",
        "stuck_value": sv,
    }


def gen_hard_sqc_fault(
    rng: random.Random, comp: Dict[str, Any], stuck_value: Optional[int]
) -> Dict[str, Any]:
    """Generates a permanent stuck-at fault descriptor for the L1 Instruction Cache (SQC)."""
    num_sqc = comp["instances"]["sqc"]
    num_sets = comp["sets"]
    assoc = comp["assoc"]
    line_size = comp.get("line_size", 64)

    sqc = rng.randrange(num_sqc)
    set_idx = rng.randrange(num_sets)
    way_idx = rng.randrange(assoc)
    byte_offset = rng.randrange(line_size)
    bit_in_byte = rng.randrange(8)

    sv = stuck_value if stuck_value is not None else rng.randrange(2)

    return {
        "location": {
            "type": "cache",
            "cache_type": "sqc",
            "sqc": sqc,
            "set": set_idx,
            "way": way_idx,
            "byte": byte_offset,
            "bit": bit_in_byte,
        },
        "mode": "stuck_at",
        "stuck_value": sv,
    }


if __name__ == "__main__":
    main()
