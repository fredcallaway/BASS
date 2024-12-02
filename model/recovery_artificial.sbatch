#!/bin/bash
#SBATCH -J bass-recovery-artificial
#SBATCH -c 48
#SBATCH --mem-per-cpu 1500 # Memory request
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%j_%A_%a.log # Combined output
#SBATCH --array=1-258

julia +1.9 -p auto --project="." run_recovery_artificial.jl $SLURM_ARRAY_TASK_ID