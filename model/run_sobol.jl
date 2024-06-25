@everywhere begin
    using Serialization
    using ProgressMeter
    using Sobol
    using SplitApplyCombine
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
end
include("sobol_search.jl")
# %% --------

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

# %% --------

t = prepare_trials(data1)[1]


# %% --------

box = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.5),
    cost = (.01, .1),
    prior_mean = (0, 1.5µ),
    prior_precision = 1 / σ^2,
)


# %% --------

# sobol_res = sobol_search(box, 5000, data1; repeats=1, dt=.1, tol=50)
# sobol_res.results
sobol_res = sobol_search(box, 5000, data1; repeats=10, dt=.01, tol=50)
serialize("tmp/apr14-soobl", sobol_res)

