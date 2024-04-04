@everywhere begin
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

function empirical_prior(data; α=1)
    µ, σ = juxt(mean, std)(flatten(data.value))
    α * µ, σ
end

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

# %% --------

box = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.5),
    cost = (.01, .1),
    prior_mean = (0, 2µ),
    prior_precision = 1 / σ^2,
)

run_sobol_group(BDDM, "choice-only", box, 1000, repeats=10, dt=.05, tol=10000.)

