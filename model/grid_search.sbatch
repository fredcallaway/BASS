#!/bin/bash
#SBATCH -J bass-grid
#SBATCH -c 32
#SBATCH --mem-per-cpu 800 # Memory request
#SBATCH -t 0-02:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%A_%a_exp1.log # %A=array job ID, %a=array task ID
#SBATCH --array=1-31

julia +1.9 -p $SLURM_CPUS_PER_TASK --project="." run_grid_search.jl $1 $SLURM_ARRAY_TASK_ID