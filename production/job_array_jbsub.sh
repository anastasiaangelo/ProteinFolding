#!/bin/bash

JOB_NAME="ProteinFolding"
NUM_TASKS=24  # Total tasks
PARALLEL_JOBS=3  # Max parallel tasks (equivalent to %3 in SLURM)
MEM_PER_TASK=50000  # Adjust memory per task if needed
CPUS_PER_TASK=3 

# Define home and job-wide run directory
HOMEDIR=$(pwd)
MAIN_JOB_ID=$(date +%s)  # Use timestamp as unique job ID
RUNDIR="$HOMEDIR/A$MAIN_JOB_ID"
mkdir -p "$RUNDIR"

# Store git commit hash (only once)
COMMIT=$(git rev-parse HEAD)
echo "Running script with commit: $COMMIT"
echo "$COMMIT" > "$RUNDIR/commit_hash.txt"

# Submit jobs in batches
for TASK_ID in $(seq 1 $NUM_TASKS); do
    # Define per-task run directory
    TASK_RUNDIR="$RUNDIR/run_$TASK_ID"
    mkdir -p "$TASK_RUNDIR"

    # Copy the script into the task folder
    cp "$HOMEDIR/simulations_array_job_lsf.py" "$TASK_RUNDIR/"

    # Submit the job using jbsub
    jbsub -q x86_24h \
          -mem $MEM_PER_TASK \
          -cores $CPUS_PER_TASK \
          -o "$TASK_RUNDIR/task_output.out" \
          -e "$TASK_RUNDIR/task_error.err" \
          "cd $TASK_RUNDIR && source /u/aag/proteinfolding/ccc_venv/bin/activate && python3 -u simulations_array_job_lsf.py \"$COMMIT\" \"$TASK_ID\""

    # Limit the number of parallel jobs
    if (( TASK_ID % PARALLEL_JOBS == 0 )); then
        wait
    fi
done