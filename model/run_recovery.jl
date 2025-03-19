using Distributed
@everywhere begin
    include("base.jl")
    include("likelihood.jl")
    using Serialization
end
include("grid_search.jl")
using SplitApplyCombine
using DataFrames, CSV
# %% --------

# version = "2024-11-20"
version = "2024-12-16"
data1 = load_human_data(1)
µ, σ = empirical_prior(data1)
group_fit = true

# fit_box = Box(
#     base_precision=(0.01, 0.1),
#     attention_factor=(0.0, 1.0),
#     cost=(0.01, 0.1),
#     prior_mean=µ,
#     prior_precision=1 / σ^2,
#     )
    
sim_box = Box(
    base_precision=0.05,
    attention_factor=(0., 1.0),
    cost=0.06,
    prior_mean=µ,
    prior_precision=1 / σ^2,
)

fit_box = sim_box

# %% --------

if isempty(ARGS) || ARGS[1] == "setup"
    params = reduce(vcat, grid(7, sim_box))
    @show length(params)
    mkpath("tmp/bddm/grid/recovery/$version/")
    serialize("tmp/bddm/grid/recovery/$version/params", params)

elseif ARGS[1] == "process"
    params = deserialize("tmp/bddm/grid/recovery/$version/params")
    mkpath("results/recovery/$version")

    path = "tmp/bddm/grid/recovery/$version/"
    flatmap(eachindex(params)) do param_id
        subjects = group_fit ? ["group"] : unique(data1.subject)
        flatmap(subjects) do subject
            x = deserialize("$path/$param_id/$subject")
            map(x.candidates[:], x.results[:]) do c, r
                (; param_id, subject, c.base_precision, c.attention_factor, c.cost, r.logp, r.std, r.converged)
            end
        end
    end |> CSV.write("results/recovery/$version/likelihoods.csv", writeheader=true)

    map(eachindex(params)) do param_id
        (;param_id, params[param_id]...)
    end |> CSV.write("results/recovery/$version/generating_params.csv", writeheader=true)

else
    params = deserialize("tmp/bddm/grid/recovery/$version/params")
    for param_id in eval(Meta.parse(ARGS[1]))
        model = BDDM(; params[param_id]...)
        sim_data = simulate_dataset(model, prepare_trials(data1));
        grid_search(BDDM, "recovery/$version/$(param_id)", fit_box, 7, sim_data; repeats=100, ε=.05, tol=5, group_fit)
    end
    # grid_search(BDDM, "recovery/$version/$(param_id)", fit_box, 7, sim_data; repeats=1, ε=.05, tol=1000)
end
