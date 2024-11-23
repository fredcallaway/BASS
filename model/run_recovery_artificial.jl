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

version = "2024-11-21"

tmp_path = "tmp/bddm/grid/recovery-artificial/$version"
results_path = "results/recovery-artificial/$version"
mkpath(tmp_path)
mkpath(results_path)

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)
fit_box = Box(
    base_precision=(0.01, 0.1),
    attention_factor=(0.5, 1.5),
    cost=(0.01, 0.1),
    prior_mean=µ,
    prior_precision=1 / σ^2,
)
sim_box = Box(
    base_precision = .055,
    attention_factor = (0.5, 1.5),
    cost = .055,
    prior_mean=µ,
    prior_precision=1 / σ^2,
)

SimTrial(t::SimTrial) = t

function simulate_data(model)
    vs = range(0, 1, length=3)[1:end-1]

    g = grid(
        v1=vs,
        v2=vs,
        order=[:longfirst, :shortfirst],
        subject=1:100
    )

    trials = map(g) do (; v1, v2, order, subject)
        HumanTrial(;
            value=[v1, v2],
            confidence=[4, 4],
            presentation_distributions=PRESENTATION_DURATIONS[order],
            real_presentation_times=Int[],
            subject,
            choice=0,
            rt=0.0,
            dt=0.025
        )
    end[:]

    simulate_dataset(model, trials)
end

if isempty(ARGS) || ARGS[1] == "setup"
    params = reduce(vcat, grid(7, sim_box))
    serialize("$tmp_path/params", params)
    @show length(params)
    println("Wrote $tmp_path/params")
elseif ARGS[1] == "process"
    params = deserialize("$tmp_path/params")

    flatmap(eachindex(params)) do param_id
        flatmap(readdir("$tmp_path/$param_id")) do subject
            x = deserialize("$tmp_path/$param_id/$subject")
            map(x.candidates[:], x.results[:]) do c, r
                (; param_id, subject, c.base_precision, c.attention_factor, c.cost, r.logp, r.std, r.converged)
            end
        end
    end |> CSV.write("$results_path/likelihoods.csv", writeheader=true)

    map(eachindex(params)) do param_id
        (;param_id, params[param_id]...)
    end |> CSV.write("$results_path/generating_params.csv", writeheader=true)
else
    params = deserialize("$tmp_path/params")
    PARAM_ID = parse(Int, ARGS[1])
    model = BDDM(; params[PARAM_ID]...)
    sim_data = simulate_data(model)
    grid_search(BDDM, "recovery-artificial/$version/$(PARAM_ID)", fit_box, 7, sim_data; repeats=10, ε=.05, tol=4)
end


