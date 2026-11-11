#!/bin/bash
set -euo pipefail

# ============================================================
# BFS L1 FAULT-INJECTION CAMPAIGN
# 100 runs per component/fault configuration
# Resume-safe: completed runs have status.txt
# ============================================================

# ----------------------------
# 1. PATH DEFINITIONS
# ----------------------------
GEM5_ROOT="${GEM5_ROOT:-$HOME/gem5}"
HEC_ROOT="${HEC_ROOT:-$HOME/HeCBench}"
EXP_ROOT="${EXP_ROOT:-$GEM5_ROOT/experiments/BFS_L1_100}"

DOCKER_EXP_ROOT="/gem5/experiments/BFS_L1_100"

mkdir -p "$EXP_ROOT"
mkdir -p "$EXP_ROOT/Benchmarks_Info"

# ----------------------------
# 2. CAMPAIGN SETTINGS
# ----------------------------
MAX_CORES=3
TIMEOUT_LIMIT="60m"
NRUNS=100
NCORES=3
VREG_FILE_SIZE=2024

# "Until L1" means:
#   include register files + L1 caches
#   exclude TCC/L2
COMPONENTS=("vgpr_file" "sgpr_file" "tcp_cache" "sqc_cache")

# If you want ONLY L1 caches, use this instead:
# COMPONENTS=("tcp_cache" "sqc_cache")

FAULT_CONFIGS=("transient_1" "hard_1" "hard_3")

B_NAME="bfs65536"
BFS_BIN="/HeCBench/src/bfs-hip/main"
BFS_INPUT="/HeCBench/rodinia-data/bfs/graph65536.txt"

# ----------------------------
# 3. BASIC CHECKS
# ----------------------------
if [ ! -x "$GEM5_ROOT/build/VEGA_X86/gem5.opt" ]; then
    echo "ERROR: gem5.opt not found or not executable:"
    echo "$GEM5_ROOT/build/VEGA_X86/gem5.opt"
    exit 1
fi

if [ ! -x "$HEC_ROOT/src/bfs-hip/main" ]; then
    echo "ERROR: BFS binary not found:"
    echo "$HEC_ROOT/src/bfs-hip/main"
    exit 1
fi

if [ ! -f "$HEC_ROOT/rodinia-data/bfs/graph65536.txt" ]; then
    echo "ERROR: BFS input not found:"
    echo "$HEC_ROOT/rodinia-data/bfs/graph65536.txt"
    exit 1
fi

docker image inspect ghcr.io/gem5/gcn-gpu:v24-0 >/dev/null 2>&1 || {
    echo "ERROR: Docker image missing: ghcr.io/gem5/gcn-gpu:v24-0"
    exit 1
}

trap 'echo "Interrupted. Finished runs have status.txt. Unfinished runs will be retried next time."; jobs -r -p | xargs -r kill 2>/dev/null || true; exit 130' INT TERM

# ----------------------------
# 4. HELPER FUNCTIONS
# ----------------------------
get_category() {
    if [[ "$1" == *"gpr"* ]]; then
        echo "Register_Files"
    else
        echo "Caches"
    fi
}

get_proper_comp_name() {
    case "$1" in
        "vgpr_file") echo "VRF" ;;
        "sgpr_file") echo "SRF" ;;
        "tcp_cache") echo "TCP" ;;
        "sqc_cache") echo "SQC" ;;
        "tcc_cache") echo "TCC" ;;
        *) echo "$1" ;;
    esac
}

decode_fault_config() {
    local F_CONF="$1"

    if [[ "$F_CONF" == "transient_1" ]]; then
        echo "transient 1 Transient 100runs_1fault"
    elif [[ "$F_CONF" == "hard_1" ]]; then
        echo "hard 1 Hard 100runs_1fault"
    elif [[ "$F_CONF" == "hard_3" ]]; then
        echo "hard 3 Hard 100runs_3faults"
    else
        echo "ERROR unknown fault config: $F_CONF"
        exit 1
    fi
}

write_summary() {
    local SUMMARY="$EXP_ROOT/campaign_summary.tsv"

    {
        echo -e "component\tfault_config\trun\tresult\texit_code"
        find "$EXP_ROOT" -path "*/runs/run_*/status.txt" | sort | while read -r SF; do
            awk -F= '
                /^DIR_COMP=/{comp=$2}
                /^F_CONF=/{fconf=$2}
                /^RUN=/{run=$2}
                /^RESULT=/{result=$2}
                /^EXIT_CODE=/{exitcode=$2}
                END {
                    if (comp != "") {
                        print comp "\t" fconf "\t" run "\t" result "\t" exitcode
                    }
                }
            ' "$SF"
        done
    } > "$SUMMARY"

    echo "Summary written to: $SUMMARY"
}

# ----------------------------
# 5. GOLDEN RUN
# ----------------------------
GOLDEN_DIR="$EXP_ROOT/Benchmarks_Info/Golden_${B_NAME}"
DOCKER_GOLDEN_DIR="$DOCKER_EXP_ROOT/Benchmarks_Info/Golden_${B_NAME}"
GOLDEN_LOG="$GOLDEN_DIR/run.log"
INFO_FILE="$EXP_ROOT/Benchmarks_Info/${B_NAME}_info.txt"

mkdir -p "$GOLDEN_DIR"

if [ ! -f "$GOLDEN_DIR/m5out/stats.txt" ] || ! grep -q "Passed" "$GOLDEN_LOG" 2>/dev/null; then
    echo "=== Running golden trace for $B_NAME ==="

    rm -rf "$GOLDEN_DIR/m5out"
    rm -f "$GOLDEN_LOG"

    timeout -s KILL "$TIMEOUT_LIMIT" docker run --rm \
        -v "$GEM5_ROOT:/gem5" \
        -v "$HEC_ROOT:/HeCBench" \
        -w /gem5 \
        ghcr.io/gem5/gcn-gpu:v24-0 \
        build/VEGA_X86/gem5.opt \
        -d "$DOCKER_GOLDEN_DIR/m5out" \
        configs/example/apu_se.py \
        -n "$NCORES" \
        --vreg-file-size "$VREG_FILE_SIZE" \
        --golden-trace \
        -c "$BFS_BIN" \
        --options="$BFS_INPUT" \
        > "$GOLDEN_LOG" 2>&1 || {
            echo "ERROR: Golden run failed or timed out."
            echo "Check: $GOLDEN_LOG"
            exit 1
        }
else
    echo "=== Golden trace already exists. Reusing it. ==="
fi

if ! grep -q "Passed" "$GOLDEN_LOG"; then
    echo "ERROR: Golden run did not show Passed."
    echo "Check: $GOLDEN_LOG"
    exit 1
fi

echo "=== Extracting profiling info ==="
{
    echo "=== $B_NAME PROFILING INFO ==="
    grep "KERNEL_START_TICK:" "$GOLDEN_LOG" || echo "KERNEL_START: NOT FOUND"
    grep "KERNEL_END_TICK:" "$GOLDEN_LOG" || echo "KERNEL_END: NOT FOUND"
    echo "--- Cache Accesses ---"
    grep "tcp_cntrl.*demand_accesses" "$GOLDEN_DIR/m5out/stats.txt" || true
    grep "sqc_cntrl.*demand_accesses" "$GOLDEN_DIR/m5out/stats.txt" || true
    echo "--- Register Usage ---"
    grep "vRF.*reads" "$GOLDEN_DIR/m5out/stats.txt" || true
    grep "sRF.*reads" "$GOLDEN_DIR/m5out/stats.txt" || true
} > "$INFO_FILE"

RAW_START=$(grep -m1 "KERNEL_START_TICK:" "$GOLDEN_LOG" | awk '{print $NF}' || true)
RAW_END=$(grep -m1 "KERNEL_END_TICK:" "$GOLDEN_LOG" | awk '{print $NF}' || true)

if [[ -z "$RAW_START" || -z "$RAW_END" || "$RAW_START" == "0" || "$RAW_END" == "0" ]]; then
    echo "ERROR: Missing kernel start/end ticks."
    echo "RAW_START=$RAW_START"
    echo "RAW_END=$RAW_END"
    echo "Check: $GOLDEN_LOG"
    exit 1
fi

FAULT_START=$((RAW_START + 1000))
FAULT_END=$((RAW_END - 1000))

if [ "$FAULT_END" -le "$FAULT_START" ]; then
    FAULT_START="$RAW_START"
    FAULT_END="$RAW_END"
fi

echo "Golden Passed."
echo "RAW_START=$RAW_START"
echo "RAW_END=$RAW_END"
echo "FAULT_START=$FAULT_START"
echo "FAULT_END=$FAULT_END"
echo "Info file: $INFO_FILE"

# ----------------------------
# 6. FAULT CAMPAIGN
# ----------------------------
TOTAL_RUNS=$(( ${#COMPONENTS[@]} * ${#FAULT_CONFIGS[@]} * NRUNS ))
echo "=== Starting BFS L1 campaign ==="
echo "Expected simulations: $TOTAL_RUNS"
echo "Experiment root: $EXP_ROOT"

for COMP in "${COMPONENTS[@]}"; do
    CAT=$(get_category "$COMP")
    DIR_COMP=$(get_proper_comp_name "$COMP")

    for F_CONF in "${FAULT_CONFIGS[@]}"; do
        read -r F_MOD F_NUM DIR_FAULT DIR_RUNS <<< "$(decode_fault_config "$F_CONF")"

        EXP_DIR="$EXP_ROOT/$CAT/$DIR_COMP/$B_NAME/$DIR_FAULT/$DIR_RUNS"
        DOCKER_EXP="$DOCKER_EXP_ROOT/$CAT/$DIR_COMP/$B_NAME/$DIR_FAULT/$DIR_RUNS"

        mkdir -p "$EXP_DIR/runs"

        echo "----------------------------------------------------"
        echo "Starting: $DIR_COMP | $B_NAME | $F_CONF"
        echo "Directory: $EXP_DIR"
        echo "----------------------------------------------------"

        for i in $(seq -w 0 $((NRUNS-1))); do
            RDIR="$EXP_DIR/runs/run_$i"
            DOCKER_RDIR="$DOCKER_EXP/runs/run_$i"
            STATUS_FILE="$RDIR/status.txt"
            MASK_FILE="$RDIR/mask.json"

            if [ -f "$STATUS_FILE" ]; then
                echo "SKIP finished: $DIR_COMP | $F_CONF | run_$i"
                continue
            fi

            mkdir -p "$RDIR"

            (
                echo "STARTED=$(date)" > "$RDIR/started.txt"
                echo "COMP=$COMP" >> "$RDIR/started.txt"
                echo "F_CONF=$F_CONF" >> "$RDIR/started.txt"
                echo "RUN=$i" >> "$RDIR/started.txt"

                if [ ! -s "$MASK_FILE" ]; then
                    docker run --rm \
                        -v "$GEM5_ROOT:/gem5" \
                        -w /gem5 \
                        ghcr.io/gem5/gcn-gpu:v24-0 \
                        python3 fault_tools/gpu_fault_gen.py \
                        --config fault_tools/gpu_config_regfiles.json \
                        --component "$COMP" \
                        --fault-model "$F_MOD" \
                        --num-sims 1 \
                        --faults-per-sim "$F_NUM" \
                        --tick-start "$FAULT_START" \
                        --tick-end "$FAULT_END" \
                        --seed $((10#$i + 10)) \
                        --output "$DOCKER_RDIR/mask.json" \
                        > "$RDIR/maskgen.log" 2>&1 || {
                            echo "MASKGEN_FAILED" > "$STATUS_FILE"
                            exit 0
                        }
                fi

                EXIT_CODE=0
                CONTAINER_NAME="sim_bfs_${DIR_COMP}_${F_CONF}_${i}_$$"

                timeout -s KILL "$TIMEOUT_LIMIT" docker run --rm \
                    --name "$CONTAINER_NAME" \
                    -v "$GEM5_ROOT:/gem5" \
                    -v "$HEC_ROOT:/HeCBench" \
                    -w /gem5 \
                    ghcr.io/gem5/gcn-gpu:v24-0 \
                    build/VEGA_X86/gem5.opt \
                    -d "$DOCKER_RDIR/m5out" \
                    configs/example/apu_se.py \
                    -n "$NCORES" \
                    --vreg-file-size "$VREG_FILE_SIZE" \
                    --fault-mask "$DOCKER_RDIR/mask.json" \
                    -c "$BFS_BIN" \
                    --options="$BFS_INPUT" \
                    > "$RDIR/run.log" 2>&1 || EXIT_CODE=$?

                if [ "$EXIT_CODE" -eq 124 ] || [ "$EXIT_CODE" -eq 137 ]; then
                    echo "HANG (Timeout)" >> "$RDIR/run.log"
                    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
                fi

                RESULT="UNKNOWN"

                if grep -q "Passed" "$RDIR/run.log"; then
                    RESULT="PASSED"
                fi

                if grep -qiE "fatal|panic|FAILED|Segmentation fault|assert" "$RDIR/run.log"; then
                    RESULT="FAIL_OR_CRASH"
                fi

                if [ "$EXIT_CODE" -eq 124 ] || [ "$EXIT_CODE" -eq 137 ]; then
                    RESULT="HANG_TIMEOUT"
                fi

                {
                    echo "EXIT_CODE=$EXIT_CODE"
                    echo "RESULT=$RESULT"
                    echo "COMP=$COMP"
                    echo "DIR_COMP=$DIR_COMP"
                    echo "F_CONF=$F_CONF"
                    echo "FAULT_MODEL=$F_MOD"
                    echo "FAULTS_PER_SIM=$F_NUM"
                    echo "RUN=$i"
                    echo "FAULT_START=$FAULT_START"
                    echo "FAULT_END=$FAULT_END"
                    echo "FINISHED=$(date)"
                } > "$STATUS_FILE"

                echo "DONE: $DIR_COMP | $F_CONF | run_$i | $RESULT"
            ) &

            while (( $(jobs -r -p | wc -l) >= MAX_CORES )); do
                wait -n || true
                docker container prune -f >/dev/null 2>&1 || true
            done
        done

        wait || true
        write_summary
    done
done

write_summary
echo "BFS L1 CAMPAIGN COMPLETE"
