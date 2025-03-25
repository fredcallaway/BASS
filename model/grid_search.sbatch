#!/bin/bash
#SBATCH -J bass-recovery-artificial
#SBATCH -c 8
#SBATCH --mem-per-cpu 800 # Memory request
#SBATCH -t 0-04:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%A_%a.log # %A=array job ID, %a=array task ID
#SBATCH --array=1-31

julia +1.9 -p 8 --project="." run_grid_search.jl 2 $SLURM_ARRAY_TASK_ID