using ProgressMeter

# %% ==================== GP likelihood ====================

@everywhere include("gp_likelihood.jl")
run_name = "bddm/sobol/v10/"
subjects = readdir("tmp/$run_name")
@showprogress "Processing sobol results" pmap(subjects) do subject
    process_sobol_result(BDDM, "v10", subject)
end

process_sobol_result(BDDM, "v11", "group")

