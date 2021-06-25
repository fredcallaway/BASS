using ProgressMeter

# %% ==================== GP likelihood ====================
version = "v13"
@everywhere include("gp_likelihood.jl")
run_name = "bddm/sobol/$version/"
subjects = readdir("tmp/$run_name")
@showprogress "Processing sobol results" pmap(subjects) do subject
    process_sobol_result(BDDM, "$version", subject)
end


