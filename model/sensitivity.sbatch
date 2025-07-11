#!/bin/bash
#SBATCH -J bass-sensitivity
#SBATCH -c 30
#SBATCH --mem-per-cpu 3000 # Memory request
#SBATCH -t 0-02:00 # Maximum execution time (D-HH:MM)
#SBATCH -o out/%A_%a_sensitivity.log # %A=array job ID, %a=array task ID
#SBATCH --array=1-6

module load R/4.2.2-fasrc01
export R_HOME="/n/sw/helmod-rocky8/apps/Core/R/4.2.2-fasrc01/lib64/R"
julia +1.9 -p $SLURM_CPUS_PER_TASK --project="." sensitivity.jl $SLURM_ARRAY_TASK_ID