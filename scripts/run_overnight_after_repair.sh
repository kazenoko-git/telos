#!/bin/bash
# Autonomous transition: Waits for Granite MMLU/ARC repair (PID 86106) to finish,
# then immediately launches the overnight Gemma 4 E4B and Phi-4 Mini pipeline.

TARGET_PID=${1:-86106}

echo "[$(date)] Waiting for Granite repair (PID $TARGET_PID) to complete..."

while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 20
done

echo "[$(date)] Granite MMLU & ARC repairs finished! Starting overnight Gemma 4 E4B pipeline..."
./.venv/bin/python3 -u scripts/run_post_school_master_pipeline.py > logs/run_post_school_master_pipeline.log 2>&1
echo "[$(date)] Overnight master pipeline completed."
