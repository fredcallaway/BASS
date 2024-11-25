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

version = "2024-11-23"

tmp_path = "tmp/bddm/grid/recovery-artificial/$version"
results_path = "results/recovery-artificial/$version"
mkpath(tmp_path)
mkpath(results_path)

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)
sim_box = fit_box = Box(
    base_precision=.05,
    attention_factor=(0.0, 1.5),
    cost=.05,
    prior_mean=µ,
    prior_precision=1 / σ^2,
)

SimTrial(t::SimTrial) = t

# %% --------

function simulate_data(model; n_subjects=100, n_values=20)
    qs = range(0, 1, length=n_values+2)[2:end-1]
    vs = quantile.(Ref(Normal(µ, σ)), qs)

    g = grid(
        v1=vs,
        v2=vs,
        order=[:longfirst, :shortfirst],
        subject=1:n_subjects
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
    data = simulate_dataset(model, trials)

    if mean(isequal(MAX_RT),getfield.(data, :rt)) > 0.1
        error("Too many timeouts")
    end

    filter!(data) do t
        t.rt < MAX_RT
    end
end

# %% --------

if isempty(ARGS) || ARGS[1] == "setup"
    params = reduce(vcat, grid(16, sim_box))
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
    @show length(sim_data)
    grid_search(BDDM, "recovery-artificial/$version/$(PARAM_ID)", fit_box, 16, sim_data; repeats=10, ε=.05, tol=4)
end


