#!/bin/bash
#SBATCH -J bass-recovery-artificial
#SBATCH -c 48
#SBATCH --mem-per-cpu 1500 # Memory request
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%j_%A_%a.log # Combined output
#SBATCH --array=1-7

julia +1.9 -p 48 --project="." run_recovery_artificial.jl $SLURM_ARRAY_TASK_ID