#!/bin/bash
#SBATCH -J bass-recovery-artificial
#SBATCH -c 128
#SBATCH --mem-per-cpu 700 # Memory request
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%j_%A_%a.out # Standard output
#SBATCH -e out/%j_%A_%a.err # Standard error
#SBATCH --array=1-7

julia +1.9 -p auto --project="." run_recovery_artificial.jl $SLURM_ARRAY_TASK_ID