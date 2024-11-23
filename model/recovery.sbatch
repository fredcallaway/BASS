#!/bin/bash
#SBATCH -J bass-recovery
#SBATCH -c 32
#SBATCH --mem-per-cpu 500 # Memory request
#SBATCH -t 0-02:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%j_%A_%a.out # Standard output
#SBATCH -e out/%j_%A_%a.err # Standard error
#SBATCH --array=1-84

julia +1.9 -p auto --project="." run_recovery.jl $SLURM_ARRAY_TASK_ID