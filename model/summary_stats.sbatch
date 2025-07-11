#!/bin/bash
#SBATCH -J bass-stats
#SBATCH -c 32
#SBATCH --mem-per-cpu 1000 # Memory request
#SBATCH -t 0-02:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%A_%a_stats.log # %A=array job ID, %a=array task ID
#SBATCH --array=1-8

julia +1.9 -p $SLURM_CPUS_PER_TASK --project="." fit_summary_stats.jl $SLURM_ARRAY_TASK_ID
