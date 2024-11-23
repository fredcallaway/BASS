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

version = "2024-11-20"
data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

box = Box(
    base_precision=(0.01, 0.1),
    attention_factor=(0.5, 1.5),
    cost=(0.01, 0.1),
    prior_mean=µ,
    prior_precision=1 / σ^2,
)

# %% --------

if isempty(ARGS) || ARGS[1] == "setup"

    params = reduce(vcat, grid(4, box))

    filter!(params) do prm
        prm.attention_factor <= 1 && prm.cost > .01
    end
    # --- adding more attention factor values ---
    full_params = reduce(vcat, grid(7, box))
    filter!(full_params) do prm
        prm.base_precision in unique(getfield.(params, :base_precision)) &&
        prm.cost in unique(getfield.(params, :cost))
    end
    setdiff!(full_params, params)
    push!(params, full_params...)
    serialize("tmp/bddm/grid/recovery/$version/params", params)
    # ----------------------------
elseif ARGS[1] == "process"
    params = deserialize("tmp/bddm/grid/recovery/$version/params")
    mkpath("results/recovery/$version")

    path = "tmp/bddm/grid/recovery/$version/"
    flatmap(eachindex(params)) do param_id
        flatmap(readdir("$path/$param_id")) do subject
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
    PARAM_ID = parse(Int, ARGS[1])
    model = BDDM(; params[PARAM_ID]...)
    sim_data = simulate_dataset(model, prepare_trials(data1; dt=.025));
    grid_search(BDDM, "recovery/$version/$(PARAM_ID)", box, 7, sim_data; repeats=10, ε=.05, tol=4)
end


