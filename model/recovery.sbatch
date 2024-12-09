#!/bin/bash
#SBATCH -J bass-recovery
#SBATCH -c 48
#SBATCH --mem-per-cpu 600 # Memory request
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%j_%A_%a.out # Standard output
#SBATCH -e out/%j_%A_%a.err # Standard error
#SBATCH --array=1-7

julia +1.9 -p 48 --project="." run_recovery.jl $SLURM_ARRAY_TASK_ID