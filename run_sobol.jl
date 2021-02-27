@everywhere 
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
    using Serialization
    using ProgressMeter
    using Sobol
    using SplitApplyCombine
end
include("sobol_search.jl")

# %% --------
box = Box(
    base_precision = (.01, 2, :log),
    attention_factor = (0, 2),
    cost = (.01, .1, :log),
    confidence_slope = (0, 1),
    prior_mean = (-1, 1),
)

# run_sobol_group(BDDM, "test", box, 10, repeats=1, dt=.05, tol=20)
run_sobol_group(BDDM, "v11", box, 5000, repeats=10, dt=.025, tol=0)
