# Code for "Considering what we know and what we don't know"

This repo contains all the code and data used to generate the results in the following paper:

https://direct.mit.edu/opmi/article/doi/10.1162/opmi.a.3/131590/Considering-What-We-Know-and-What-We-Don-t-KnowRomy 

## Reproducing results

### Dependencies

Install ([juliaup](https://github.com/JuliaLang/juliaup))

This code requires version 1.9

    juliaup add 1.9

Install packages

    julia +1.9 --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"

### Generate summary statistics

This is only necessary if you want to reproduce the fitting or sensitivity analyses. It generates results/2025-05-06/summary_stats

    julia +1.9 -p auto --project=. generate_summary_stats.jl


This will take several hours on a laptop (maybe 8?). You can also run it on a cluster with

    sbatch generate_summary_stats.sbatch

Note that we use RCall for running regressions (bad choice, but oh well). You'll need to make sure you have an R available wherever you run this.

### Fitting and sensitivity analysis

This produces results/2025-05-06/parameter_table.tex and results/2025-05-06/summary_fits.csv

    julia +1.9 --project=. process_summary_stats.jl

### Simulate the best-fitting model (and lesioned versions)

    julia +1.9 --project=. generate_simulations.jl

## Other key files

- model.jl defines the core of the model
- dc.jl defines the directed-cognition stopping rule
- data.jl loads and preprocesses the experimental data
- regressions.jl defines the summary statistics used for fitting and sensitivity analysis
