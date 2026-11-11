# gem5 AMD GPU Microarchitectural Fault Injection Framework

A timing-detailed microarchitectural fault injection and fault-lifecycle tracking framework for **AMD GCN/Vega (`gfx900` / `VEGA_X86`) GPUs**, built on top of the **gem5** simulator and the **Ruby (`GPU_VIPER`)** memory system.

This repository contains the official implementation, fault generation tools (`fault_tools/`), and campaign automation scripts developed at the **Computer Engineering & Informatics Department (CEID), University of Patras**:
> **"Reliability Study of Microarchitectural Structures in Modern AMD GPUs"**  
> *Chrysostomos Thalis Tsionas*  
> Supervisor: Prof. George Papadimitriou

---

## Table of Contents
1. [Overview & Supported Hardware Structures](#1-overview--supported-hardware-structures)
2. [System Requirements & Environment Setup](#2-system-requirements--environment-setup)
3. [Installing the Fault Injection Extensions & Building gem5](#3-installing-the-fault-injection-extensions--building-gem5)
4. [Quick Start: Baseline Verification & Golden Trace Profiling](#4-quick-start-baseline-verification--golden-trace-profiling)
5. [Generating Fault Masks (`fault_tools/gpu_fault_gen.py`)](#5-generating-fault-masks-fault_toolsgpu_fault_genpy)
6. [Running Single Fault Injections & Lifecycle Tracking](#6-running-single-fault-injections--lifecycle-tracking)
7. [Automated Mass Fault Injection Campaigns](#7-automated-mass-fault-injection-campaigns)
8. [Internal Architecture & 64-Bit Bit-Packing Reference](#8-internal-architecture--64-bit-bit-packing-reference)
9. [License and Attribution](#9-license-and-attribution)
10. [Citation](#10-citation)

---

## 1. Overview & Supported Hardware Structures

Modern GPGPU workloads rely on massive microarchitectural state distributed across register files and cache hierarchies. This framework enables cycle-accurate Statistical Fault Injection (SFI) and deep microarchitectural fault-lifecycle analysis in gem5's Syscall Emulation (`SE`) mode across four primary GPU structures (plus experimental L2 support):

| Component Key | Microarchitectural Structure | Baseline Configuration (`gpu_config_regfiles.json` / `apu_se.py`) | Injectable Capacity |
| :--- | :--- | :--- | :--- |
| `vgpr_file` | **Vector Register File (VRF)** | 4 CUs, 4 SIMDs/CU, 2,048 VGPRs/SIMD, 64 lanes $\times$ 32b | $67,108,864$ bits ($97.3\%$) |
| `sgpr_file` | **Scalar Register File (SRF)** | 4 CUs, 4 SIMDs/CU, 2,048 SGPRs/SIMD, 32b scalar | $1,048,576$ bits ($1.5\%$) |
| `tcp_cache` | **L1 Vector Data Cache (TCP)** | Private 16 KB per CU (16 sets, 16-way, 64B lines) | $524,288$ bits ($0.76\%$) |
| `sqc_cache` | **L1 Instruction Cache (SQC)** | Shared 32 KB across 4 CUs (64 sets, 8-way, 64B lines) | $262,144$ bits ($0.38\%$) |
| `tcc_cache` | **L2 Texture Cache (TCC)** | Unified shared L2 cache (256 sets, 16-way, 64B lines) | *Experimental support* |

### Supported Fault Models
* **Transient Faults (`--fault-model transient`):** Single-bit (and multi-bit) upsets dynamically injected via gem5's event queue at a simulation tick sampled from a bounded Gaussian distribution over the profiled GPU kernel execution window $[T_{\text{start}}, T_{\text{end}})$.
* **Permanent Stuck-at Faults (`--fault-model hard`):** Persistent **Stuck-at-0 (`SA0`)** and **Stuck-at-1 (`SA1`)** hardware defects enforced dynamically via read-intercepts whenever the targeted storage location is accessed.

### Fault Lifecycle & Outcome Telemetry
Beyond macro-outcome classification (**Masked**, **Crash**, **Timeout**, **Silent Data Corruption - SDC**) and **Functional / Structure-Weighted AVF ($wAVF$)**, the C++ backend tracks:
* **Activation (`WIDTH`):** Total target accesses, dormant (`no_effect`) accesses, and effective activations (respecting wavefront `execMask` and cache packet byte boundaries).
* **Causal Provenance Propagation (`DEPTH`):** Multi-generation taint tracking ($0, 1, 2, 3+$ generations) across destination register writes, including dynamic register-reuse clearing (`TAINT_CLEAR`) and the **SRF $\rightarrow$ VRF scalar broadcast bridge**.
* **Execution Fan-out (`WIDTH`):** Spatial exposure classification across lanes, wavefronts, SIMDs, and Compute Units (`SINGLE_LANE` / `SINGLE_WAVEFRONT` up to `MULTI_CU`).

---

## 2. System Requirements & Environment Setup

To guarantee 100% reproducibility without host compiler or ROCm driver conflicts, all builds and simulations run inside the official **gem5 GCN/Vega GPU Docker container** (`ghcr.io/gem5/gcn-gpu:v24-0`), which provides Ubuntu 20.04, ROCm 4.0.1 (`hipcc`), Python 3, and SCons.

### Pull the Docker Image
```bash
docker pull ghcr.io/gem5/gcn-gpu:v24-0
```
*(Note: On HPC clusters using Singularity/Apptainer, this image can be pulled directly via `singularity pull docker://ghcr.io/gem5/gcn-gpu:v24-0`).*

### Clone Baseline `gem5` and `gem5-resources`
Create the base directories in your home folder (`$HOME/gem5` and `$HOME/gem5-resources`):

```bash
cd $HOME

# Clone gem5 and checkout the baseline revision
git clone https://github.com/gem5/gem5.git gem5
cd gem5
git checkout 1052375c41873da8dab1c5aace05e07708085

# Clone gem5-resources (contains GPU benchmarks & test workloads)
cd $HOME
git clone https://github.com/gem5/gem5-resources.git gem5-resources
```

---

## 3. Installing the Fault Injection Extensions & Building gem5

### Overlay the Fault Injection Files into `$HOME/gem5`
If you are applying the framework on top of a clean `gem5` checkout, copy the `fault_tools/` directory and the modified simulator files into their exact target paths inside `$HOME/gem5`:

| Repository Path | Target Path in `$HOME/gem5` | Role in the Fault Injection Toolchain |
| :--- | :--- | :--- |
| `fault_tools/gpu_fault_gen.py` | `fault_tools/gpu_fault_gen.py` | Phase 1: Deterministic JSON fault mask generator (Uniform spatial + Gaussian temporal). |
| `fault_tools/gpu_config_regfiles.json` | `fault_tools/gpu_config_regfiles.json` | Hardware geometry definitions for `vgpr_file`, `sgpr_file`, `tcp_cache`, `sqc_cache`, `tcc_cache`. |
| `configs/example/apu_se.py` | `configs/example/apu_se.py` | Phase 2: Parses `--fault-mask`, `--fault-mask-id`, `--propagation`, `--golden-trace` & performs 64-bit bit-packing. |
| `src/gpu-compute/GPU.py` | `src/gpu-compute/GPU.py` | Defines `Shader` SimObject `VectorParam.UInt64` fault lists, tracking flags, and `GPUDispatcher.golden_trace`. |
| `src/gpu-compute/shader.hh` | `src/gpu-compute/shader.hh` | Declares unpacked fault structs, `Inject*FaultEvent` classes, and provenance ID allocator. |
| `src/gpu-compute/shader.cc` | `src/gpu-compute/shader.cc` | Phase 3: Unpacks 64-bit fault payloads in `Shader::startup()`, schedules events, and installs hard faults. |
| `src/gpu-compute/vector_register_file.hh` | `src/gpu-compute/vector_register_file.hh` | VRF `HardKey` map, fault injection declarations, and per-lane `PropagationTag` state. |
| `src/gpu-compute/vector_register_file.cc` | `src/gpu-compute/vector_register_file.cc` | VRF `injectBitFlip()`, `readSource()` stuck-at intercept, width exposure & causal depth tracking. |
| `src/gpu-compute/scalar_register_file.hh` | `src/gpu-compute/scalar_register_file.hh` | SRF `hardFaultMap`, fault injection declarations, and scalar `PropagationTag` state. |
| `src/gpu-compute/scalar_register_file.cc` | `src/gpu-compute/scalar_register_file.cc` | SRF `injectBitFlip()`, `readSource()` stuck-at intercept, width exposure & causal depth tracking. |
| `src/mem/ruby/system/GPUCoalescer.hh` | `src/mem/ruby/system/GPUCoalescer.hh` | L1 TCP fault forwarding methods and `exitCallback()` declaration. |
| `src/mem/ruby/system/GPUCoalescer.cc` | `src/mem/ruby/system/GPUCoalescer.cc` | L1 TCP load fault interception (`readCallback`), packet poison tagging, and lane consumer tracking. |
| `src/mem/ruby/system/Sequencer.hh` | `src/mem/ruby/system/Sequencer.hh` | L1 SQC fault forwarding methods and `exitCallback()` declaration. |
| `src/mem/ruby/system/Sequencer.cc` | `src/mem/ruby/system/Sequencer.cc` | L1 SQC `IFETCH` fault interception (`hitCallback`), packet poison tagging, and wavefront consumer tracking. |

*(Ensure all remaining modified files under `src/gpu-compute/` and `src/mem/ruby/` are also copied into their corresponding directories).*

### Build / Rebuild `VEGA_X86/gem5.opt`
Compile the `VEGA_X86` simulator binary inside the Docker container:

```bash
docker run --rm -v $HOME/gem5:/gem5 -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 scons build/VEGA_X86/gem5.opt -j$(nproc)
```

---

## 4. Quick Start: Baseline Verification & Golden Trace Profiling

### Compile the `square` Test Benchmark
```bash
docker run --rm -v $HOME/gem5-resources:/gem5-resources -w /gem5-resources/src/gpu/square ghcr.io/gem5/gcn-gpu:v24-0 make
```

### Run a Standard Fault-Free Simulation
Verify that your compiled `build/VEGA_X86/gem5.opt` runs `square` cleanly without fault injection:

```bash
docker run --rm -v $HOME/gem5:/gem5 -v $HOME/gem5-resources:/gem5-resources -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 build/VEGA_X86/gem5.opt configs/example/apu_se.py -n 3 -c /gem5-resources/src/gpu/square/bin/square
```

### Run a Golden Trace (`--golden-trace`)
Before running a **transient** fault injection campaign on any workload, execute a fault-free Golden Run with `--golden-trace`. This prints `KERNEL_START_TICK:` and `KERNEL_END_TICK:` to the log so you can bound transient fault sampling strictly to the active GPU execution phase:

```bash
mkdir -p $HOME/gem5/golden_square

docker run --rm \
  -v $HOME/gem5:/gem5 \
  -v $HOME/gem5-resources:/gem5-resources \
  -w /gem5 \
  ghcr.io/gem5/gcn-gpu:v24-0 \
  build/VEGA_X86/gem5.opt \
  -d /gem5/golden_square/m5out \
  configs/example/apu_se.py \
  -n 3 \
  --vreg-file-size 2048 \
  --golden-trace \
  -c /gem5-resources/src/gpu/square/bin/square > $HOME/gem5/golden_square/run.log 2>&1

grep "KERNEL_START_TICK:" $HOME/gem5/golden_square/run.log
grep "KERNEL_END_TICK:" $HOME/gem5/golden_square/run.log
```

---

## 5. Generating Fault Masks (`fault_tools/gpu_fault_gen.py`)

The generator script `fault_tools/gpu_fault_gen.py` creates a JSON file containing deterministic fault masks. It uses the hardware geometry configuration file `fault_tools/gpu_config_regfiles.json` to determine valid coordinate bounds (defaulting to `2048` physical VGPRs and SGPRs per SIMD, which can be customized if needed).

### `fault_tools/gpu_config_regfiles.json`
```json
{
  "components": {
    "vgpr_file": {
      "type": "regfile",
      "instances": { "cu": 4, "simd": 4 },
      "num_regs": 2048,
      "lanes_per_reg": 64,
      "bits_per_lane": 32
    },
    "sgpr_file": {
      "type": "regfile",
      "instances": { "cu": 4, "simd": 4 },
      "num_regs": 2048,
      "bits_per_reg": 32
    },
    "tcp_cache": {
      "type": "cache",
      "instances": { "cu": 4 },
      "sets": 16,
      "assoc": 16,
      "line_size": 64
    },
    "sqc_cache": {
      "type": "cache",
      "instances": { "sqc": 1 },
      "sets": 64,
      "assoc": 8,
      "line_size": 64
    },
    "tcc_cache": {
      "type": "cache",
      "instances": { "tcc": 1 },
      "sets": 256,
      "assoc": 16,
      "line_size": 64
    }
  }
}
```

### CLI Options for `gpu_fault_gen.py`
| Option | Values / Type | Description |
| :--- | :--- | :--- |
| `--config` | `str` (Path) | Path to the GPU geometry JSON file (`fault_tools/gpu_config_regfiles.json`). |
| `--component` | `vgpr_file`, `sgpr_file`, `tcp_cache`, `sqc_cache`, `tcc_cache` | Target microarchitectural structure to inject faults into. |
| `--fault-model` | `transient` \| `hard` | `transient` for timed bit-flips; `hard` for permanent stuck-at constraints. |
| `--num-sims` | `int` | Number of simulation mask entries (`id: 0 .. num_sims-1`) to generate in the output JSON. |
| `--faults-per-sim` | `int` | Number of faults per simulation run (`1` for single-bit campaigns, or `>1` for multi-bit). |
| `--tick-start` | `int` | *(Transient only)* Start tick of the GPU kernel execution window (inclusive). |
| `--tick-end` | `int` | *(Transient only)* End tick of the GPU kernel execution window (exclusive, **required** for transient). |
| `--sigma-frac` | `float` (Default: `6.0`) | *(Transient only)* Gaussian $\sigma = (\text{tick\_end} - \text{tick\_start}) / \text{sigma\_frac}$. |
| `--stuck-value` | `0` \| `1` (Optional) | *(Hard only)* `0` for Stuck-at-0 (`SA0`), `1` for Stuck-at-1 (`SA1`). If omitted, chosen randomly per fault. |
| `--seed` | `int` (Default: `1`) | Base deterministic RNG seed (run `sim_id` uses `seed + sim_id`). |
| `--output` | `str` (Path) | Output JSON mask file path. |

### Example Mask Generation Commands

```bash
# Generate a Transient VRF Fault Mask (using profiled [tick-start, tick-end))
docker run --rm -v $HOME/gem5:/gem5 -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 \
  python3 fault_tools/gpu_fault_gen.py \
  --config fault_tools/gpu_config_regfiles.json \
  --component vgpr_file \
  --fault-model transient \
  --num-sims 1000 \
  --faults-per-sim 1 \
  --tick-start 100000 \
  --tick-end 500000 \
  --seed 1 \
  --output /gem5/masks_vrf_transient.json

# Generate a Permanent Stuck-at-0 (SA0) Mask for the L1 Data Cache (TCP)
docker run --rm -v $HOME/gem5:/gem5 -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 \
  python3 fault_tools/gpu_fault_gen.py \
  --config fault_tools/gpu_config_regfiles.json \
  --component tcp_cache \
  --fault-model hard \
  --stuck-value 0 \
  --num-sims 1000 \
  --faults-per-sim 1 \
  --seed 1 \
  --output /gem5/masks_tcp_sa0.json

# Generate a Permanent Stuck-at-1 (SA1) Mask for the Scalar Register File (SRF)
docker run --rm -v $HOME/gem5:/gem5 -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 \
  python3 fault_tools/gpu_fault_gen.py \
  --config fault_tools/gpu_config_regfiles.json \
  --component sgpr_file \
  --fault-model hard \
  --stuck-value 1 \
  --num-sims 1000 \
  --faults-per-sim 1 \
  --seed 1 \
  --output /gem5/masks_srf_sa1.json
```

---

## 6. Running Single Fault Injections & Lifecycle Tracking

Once a mask file is generated, pass `--fault-mask` and `--fault-mask-id` (and optionally `--propagation`) to `configs/example/apu_se.py`:

### Custom Fault Injection Flags in `apu_se.py`
* `--fault-mask <path>`: Path to the JSON fault mask file.
* `--fault-mask-id <int>`: Which simulation `id` from the JSON mask array to execute (defaults to `0`).
* `--propagation {none,vrf,srf,tcp,sqc,all}`: Enables microarchitectural fault-lifecycle tracking.
  * `vrf` / `srf`: Enables both **WIDTH** (target access, effective activation, execution fan-out) and **DEPTH** (multi-generation causal register provenance).
  * `tcp` / `sqc`: Enables **WIDTH** tracking at the direct cache-consumer boundary.
* `--propagation-verbose`: Emits per-instruction `VRF_PROPAGATION`, `SRF_PROPAGATION`, and `FAULT_TRACE` events in addition to the end-of-run `FAULT_ANALYSIS` summary.

### Example: Run Simulation ID `0` from `masks_tcp_sa0.json` with TCP Lifecycle Tracking
```bash
docker run --rm \
  -v $HOME/gem5:/gem5 \
  -v $HOME/gem5-resources:/gem5-resources \
  -w /gem5 \
  ghcr.io/gem5/gcn-gpu:v24-0 \
  build/VEGA_X86/gem5.opt \
  -d /gem5/m5out_tcp_sa0_id0 \
  configs/example/apu_se.py \
  -n 3 \
  --vreg-file-size 2048 \
  --fault-mask /gem5/masks_tcp_sa0.json \
  --fault-mask-id 0 \
  --propagation tcp \
  -c /gem5-resources/src/gpu/square/bin/square
```

---

## 7. Automated Mass Fault Injection Campaigns

For large-scale statistical campaigns, the framework uses automated Bash orchestration scripts that perform:
1. **Profiling Phase (Golden Runs):** Runs each benchmark with `--golden-trace`, saves the golden `m5out/stats.txt`, and extracts `KERNEL_START_TICK:` and `KERNEL_END_TICK:`.
2. **Mass Injection Phase:** Computes the safe injection window (`FAULT_START` to `FAULT_END`), generates per-run deterministic masks (`--seed $((10#$i + 10))`), launches up to `MAX_CORES` concurrent Docker containers, and enforces a host-level `timeout` guard to detect and clean up hanging simulations (`Timeout` outcomes).

### Example Campaign Script (`campaign.sh`)
```bash
#!/bin/bash
set -euo pipefail

# ==========================================
# 1. PATH DEFINITIONS
# ==========================================
GEM5_ROOT="$HOME/gem5"
RES_ROOT="$HOME/gem5-resources"
MASS_EXP_DIR="$GEM5_ROOT/experiments/mass_experiments"
BENCH_INFO_DIR="$MASS_EXP_DIR/benchmarks_info"

mkdir -p "$MASS_EXP_DIR"
mkdir -p "$BENCH_INFO_DIR"

# ==========================================
# 2. CAMPAIGN SETTINGS
# ==========================================
MAX_CORES=3
TIMEOUT_LIMIT="60m" # Safely increased to 1 hour
NRUNS=500

COMPONENTS=("vgpr_file" "sgpr_file" "tcp_cache" "sqc_cache" "tcc_cache")
FAULT_CONFIGS=("transient_1" "hard_1" "hard_3")

# BENCHMARKS
declare -A BENCHMARKS
BENCHMARKS=(
    ["square_expected"]="/gem5-resources/src/gpu/square/bin/square"
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
get_category() {
    if [[ "$1" == *"gpr"* ]]; then echo "Register_Files"; else echo "Caches"; fi
}

get_proper_comp_name() {
    case $1 in
        "vgpr_file") echo "VRF" ;; "sgpr_file") echo "SRF" ;;
        "tcp_cache") echo "TCP" ;; "sqc_cache") echo "SQC" ;;
        "tcc_cache") echo "TCC" ;; *) echo "$1" ;;
    esac
}

# ==========================================
# 4. PROFILING PHASE (Golden Runs & Info Extraction)
# ==========================================
echo "=== Phase 1: Profiling & Golden Runs ==="
for B_NAME in "${!BENCHMARKS[@]}"; do
    B_PATH=${BENCHMARKS[$B_NAME]}
    GOLDEN_DIR="$MASS_EXP_DIR/benchmarks_info/Golden_${B_NAME}"
    INFO_FILE="$BENCH_INFO_DIR/${B_NAME}_info.txt"

    if [ ! -f "$GOLDEN_DIR/m5out/stats.txt" ]; then
        echo "Running Golden Trace for: $B_NAME"
        mkdir -p "$GOLDEN_DIR"
        docker run --rm -v "$GEM5_ROOT:/gem5" -v "$RES_ROOT:/gem5-resources" -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 \
          build/VEGA_X86/gem5.opt -d "/gem5/experiments/mass_experiments/benchmarks_info/Golden_${B_NAME}/m5out" \
          configs/example/apu_se.py -n 3 --vreg-file-size 2048 --golden-trace -c $B_PATH > "$GOLDEN_DIR/run.log" 2>&1
    fi

    if [ ! -f "$INFO_FILE" ]; then
        echo "Extracting stats for $B_NAME -> $INFO_FILE"
        echo "=== $B_NAME PROFILING INFO ===" > "$INFO_FILE"
        grep "KERNEL_START_TICK:" "$GOLDEN_DIR/run.log" || echo "KERNEL_START: NOT FOUND" >> "$INFO_FILE"
        grep "KERNEL_END_TICK:" "$GOLDEN_DIR/run.log" || echo "KERNEL_END: NOT FOUND" >> "$INFO_FILE"
        echo "--- Cache Accesses ---" >> "$INFO_FILE"
        grep "tcp_cntrl.*demand_accesses" "$GOLDEN_DIR/m5out/stats.txt" >> "$INFO_FILE" || true
        grep "tcc_cntrl.*demand_accesses" "$GOLDEN_DIR/m5out/stats.txt" >> "$INFO_FILE" || true
        grep "sqc_cntrl.*demand_accesses" "$GOLDEN_DIR/m5out/stats.txt" >> "$INFO_FILE" || true
        echo "--- Register Usage ---" >> "$INFO_FILE"
        grep "vRF.*reads" "$GOLDEN_DIR/m5out/stats.txt" >> "$INFO_FILE" || true
        grep "sRF.*reads" "$GOLDEN_DIR/m5out/stats.txt" >> "$INFO_FILE" || true
    fi
done

# ==========================================
# 5. MASS INJECTION CAMPAIGN
# ==========================================
echo "=== Phase 2: Fault Injection Campaign ==="
for B_NAME in "${!BENCHMARKS[@]}"; do
    B_PATH=${BENCHMARKS[$B_NAME]}
    GOLDEN_DIR="$MASS_EXP_DIR/benchmarks_info/Golden_${B_NAME}"
    
    RAW_START=$(grep -m1 "KERNEL_START_TICK:" "$GOLDEN_DIR/run.log" | awk '{print $NF}' || echo "0")
    RAW_END=$(grep -m1 "KERNEL_END_TICK:" "$GOLDEN_DIR/run.log" | awk '{print $NF}' || echo "0")
    if [ "$RAW_START" == "0" ] || [ "$RAW_END" == "0" ]; then echo "SKIPPING $B_NAME - Missing Kernel Ticks!"; continue; fi

    for COMP in "${COMPONENTS[@]}"; do
        CAT=$(get_category "$COMP"); DIR_COMP=$(get_proper_comp_name "$COMP")

        if [ "$COMP" == "tcc_cache" ]; then
            FAULT_START=$((RAW_START - 10000000000))
            [ "$FAULT_START" -lt 0 ] && FAULT_START=0
            FAULT_END=$((RAW_END + 10000000000))
        else
            FAULT_START=$((RAW_START + 1000)); FAULT_END=$((RAW_END - 1000))
        fi

        for F_CONF in "${FAULT_CONFIGS[@]}"; do
            if [[ "$F_CONF" == "transient_1" ]]; then F_MOD="transient"; F_NUM=1; DIR_FAULT="Transient"; DIR_RUNS="500runs_1fault"
            elif [[ "$F_CONF" == "hard_1" ]]; then F_MOD="hard"; F_NUM=1; DIR_FAULT="Hard"; DIR_RUNS="500runs_1fault"
            elif [[ "$F_CONF" == "hard_3" ]]; then F_MOD="hard"; F_NUM=3; DIR_FAULT="Hard"; DIR_RUNS="500runs_3faults"; fi

            EXP_DIR="$MASS_EXP_DIR/$CAT/$DIR_COMP/$B_NAME/$DIR_FAULT/$DIR_RUNS"
            DOCKER_EXP="/gem5/experiments/mass_experiments/$CAT/$DIR_COMP/$B_NAME/$DIR_FAULT/$DIR_RUNS"
            mkdir -p "$EXP_DIR/runs"

            echo "Starting: $DIR_COMP | $B_NAME | $F_CONF"
            for i in $(seq -w 0 $((NRUNS-1))); do
                RDIR="$EXP_DIR/runs/run_$i"
                if [ -f "$RDIR/run.log" ]; then continue; fi
                mkdir -p "$RDIR"

                (
                    docker run --rm -v "$GEM5_ROOT:/gem5" -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 python3 fault_tools/gpu_fault_gen.py \
                        --config fault_tools/gpu_config_regfiles.json --component "$COMP" --fault-model "$F_MOD" \
                        --num-sims 1 --faults-per-sim "$F_NUM" --tick-start "$FAULT_START" --tick-end "$FAULT_END" \
                        --seed $((10#$i + 10)) --output "$DOCKER_EXP/runs/run_$i/mask.json" > "$RDIR/maskgen.log" 2>&1

                    EXIT_CODE=0
                    timeout -s KILL $TIMEOUT_LIMIT docker run --rm --name "sim_${DIR_COMP}_${B_NAME}_${F_CONF}_$i" \
                      -v "$GEM5_ROOT:/gem5" -v "$RES_ROOT:/gem5-resources" -w /gem5 ghcr.io/gem5/gcn-gpu:v24-0 \
                      build/VEGA_X86/gem5.opt -d "$DOCKER_EXP/runs/run_$i/m5out" configs/example/apu_se.py -n 3 --vreg-file-size 2048 \
                      --fault-mask "$DOCKER_EXP/runs/run_$i/mask.json" -c $B_PATH > "$RDIR/run.log" 2>&1 || EXIT_CODE=$?

                    if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
                        echo "HANG (Timeout)" >> "$RDIR/run.log"
                        docker rm -f "sim_${DIR_COMP}_${B_NAME}_${F_CONF}_$i" > /dev/null 2>&1 || true
                    fi
                ) &

                if (( $(jobs -r -p | wc -l) >= MAX_CORES )); then wait -n; docker container prune -f > /dev/null 2>&1 || true; fi
            done
            wait
        done
    done
done
echo "=== CAMPAIGN COMPLETE ==="
```

---

## 8. Internal Architecture & 64-Bit Bit-Packing Reference

To avoid dictionary serialization overhead across the Python/C++ boundary during simulator instantiation, `apu_se.py` encodes each fault into a single 64-bit unsigned integer (`VectorParam.UInt64`), which is unpacked by `Shader::startup()` in `src/gpu-compute/shader.cc`:

### Transient Fault Bit-Packing Layouts (LSB $\rightarrow$ MSB)
| Component | SimObject VectorParam | Bit-Field Layout (from Bit 0 upwards) | Upper Bits |
| :--- | :--- | :--- | :--- |
| **VRF** | `shader.vrf_faults` | `laneBit [4:0]` (5b) \| `lane [10:5]` (6b) \| `physReg [21:11]` (11b) \| `simd [23:22]` (2b) \| `cu [25:24]` (2b) | `tick` (`>> 26`) |
| **SRF** | `shader.srf_faults` | `bit [4:0]` (5b) \| `physReg [15:5]` (11b) \| `simd [17:16]` (2b) \| `cu [19:18]` (2b) | `tick` (`>> 20`) |
| **TCP** | `shader.tcp_faults` | `bit [2:0]` (3b) \| `byte [8:3]` (6b) \| `way [12:9]` (4b) \| `set [16:13]` (4b) \| `cu [20:17]` (4b) | `tick` (`>> 21`) |
| **SQC** | `shader.sqc_faults` | `bit [2:0]` (3b) \| `byte [8:3]` (6b) \| `way [11:9]` (3b) \| `set [17:12]` (6b) \| `sqc [21:18]` (4b) | `tick` (`>> 22`) |
| **TCC** | `shader.tcc_faults` | `bit [2:0]` (3b) \| `byte [8:3]` (6b) \| `way [12:9]` (4b) \| `set [20:13]` (8b) \| `tcc [24:21]` (4b) | `tick` (`>> 25`) |

### Permanent (`hard` / Stuck-At) Fault Bit-Packing Layouts (LSB $\rightarrow$ MSB)
| Component | SimObject VectorParam | Bit-Field Layout (from Bit 0 upwards) |
| :--- | :--- | :--- |
| **VRF Hard** | `shader.vrf_hard_faults` | `stuck [0]` (1b) \| `laneBit [5:1]` (5b) \| `lane [11:6]` (6b) \| `physReg [22:12]` (11b) \| `simd [24:23]` (2b) \| `cu [26:25]` (2b) |
| **SRF Hard** | `shader.srf_hard_faults` | `stuck [0]` (1b) \| `bit [5:1]` (5b) \| `physReg [16:6]` (11b) \| `simd [18:17]` (2b) \| `cu [20:19]` (2b) |
| **TCP Hard** | `shader.tcp_hard_faults` | `stuck [0]` (1b) \| `bit [3:1]` (3b) \| `byte [9:4]` (6b) \| `way [13:10]` (4b) \| `set [17:14]` (4b) \| `cu [21:18]` (4b) |
| **SQC Hard** | `shader.sqc_hard_faults` | `stuck [0]` (1b) \| `bit [3:1]` (3b) \| `byte [9:4]` (6b) \| `way [12:10]` (3b) \| `set [18:13]` (6b) \| `sqc [22:19]` (4b) |
| **TCC Hard** | `shader.tcc_hard_faults` | `stuck [0]` (1b) \| `bit [3:1]` (3b) \| `byte [9:4]` (6b) \| `way [13:10]` (4b) \| `set [21:14]` (8b) \| `tcc [25:22]` (4b) |

---

## 9. License and Attribution

Copyright (c) 2026 Chrysostomos Thalis Tsionas
All rights reserved. 

gem5-GPUFI is released under the BSD 3-Clause License,
subject to the licenses of any incorporated third-party code.

The original fault-injection extensions were developed by
Chrysostomos Thalis Tsionas as part of a diploma thesis conducted
at the Technology and Computer Architecture Laboratory
(TCAL), University of Patras.

The project builds upon the gem5 simulator
(https://www.gem5.org/).

The original extensions are authored and maintained by
Chrysostomos Thalis Tsionas. The project is distributed as-is,
without warranties, and is not an official gem5 release.

The University of Patras and TCAL are not responsible
for providing maintenance or technical support.

---

## 10. Citation

If you use this fault-injection framework or methodology in your research, please cite:

```bibtex
@mastersthesis{Tsionas2026GPUReliability,
  author  = {Chrysostomos Thalis Tsionas},
  title   = {Reliability Study of Microarchitectural Structures in Modern AMD GPUs},
  school  = {Computer Engineering and Informatics Department (CEID), University of Patras},
  year    = {2026},
  month   = {September},
  type    = {Diploma Thesis}
}
```