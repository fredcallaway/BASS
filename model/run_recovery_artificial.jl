using Distributed
@everywhere begin
    include("base.jl")
    include("likelihood.jl")
    include("grid_search.jl")
    using Serialization
end
using SplitApplyCombine
using DataFrames, CSV
using ProgressMeter
ProgressMeter.ncalls(::typeof(flatmap), ::Any, xs::Any) = length(xs)

# %% --------

# version = "recovery-artificial-rapid/2024-11-25"
version = "recovery-artificial-rapid/2024-03-19"
tmp_path = "tmp/bddm/grid/$version"
results_path = "results/$version"
mkpath(tmp_path)
mkpath(results_path)

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

sim_box = fit_box = Box(
    base_precision = .05,
    attention_factor=(0., 1.),
    cost=0.06,
    prior_mean=µ,
    prior_precision=1 / σ^2,
)
# sim_box = fit_box = Box(
#     base_precision=(0.01, 0.1),
#     attention_factor=(0., 1.5),
#     cost=(0.01, 0.1),
#     prior_mean=µ,
#     prior_precision=1 / σ^2,
# )

SimTrial(t::SimTrial) = t

ALT_DURATIONS = Dict(
    :shortfirst => [Normal(.025, 0), Normal(.05, 0)],
    :longfirst => [Normal(.05, 0), Normal(.025, 0)]
)
function simulate_data(model; n_subjects=96, n_values=20)
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
            presentation_distributions=ALT_DURATIONS[order],
            real_presentation_times=Int[],
            subject,
            choice=0,
            rt=0.0,
            dt=.025
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




if isempty(ARGS) || ARGS[1] == "setup"
    params = reduce(vcat, grid(7, sim_box))
    filter!(p -> p.cost > 0.01, params)
    serialize("$tmp_path/params", params)
    @show length(params)
    println("Wrote $tmp_path/params")
elseif ARGS[1] == "process"
    params = deserialize("$tmp_path/params")

    param_ids = filter(eachindex(params)) do param_id
        isdir("$tmp_path/$param_id")
    end
    if length(param_ids) < length(params)
        println("Some params unavailable: $(setdiff(eachindex(params), param_ids))")
    end
    res = @showprogress flatmap(param_ids) do param_id
        flatmap(readdir("$tmp_path/$param_id")) do subject
            x = deserialize("$tmp_path/$param_id/$subject")
            map(x.candidates[:], x.results[:]) do c, r
                (; param_id, subject, c.base_precision, c.attention_factor, c.cost, r.logp, r.std, r.converged)
            end
        end
    end
    CSV.write("$results_path/likelihoods.csv", res, writeheader=true)

    map(eachindex(params)) do param_id
        (;param_id, params[param_id]...)
    end |> CSV.write("$results_path/generating_params.csv", writeheader=true)
else
    params = deserialize("$tmp_path/params")
    PARAM_ID = parse(Int, ARGS[1])
    model = BDDM(; params[PARAM_ID]...)
    sim_data = simulate_data(model)
    @show length(sim_data)
    grid_search(BDDM, "$version/$(PARAM_ID)", fit_box, 7, sim_data;
        repeats=10, ε=.05, tol=4, parallelize=:subjects
    )
end
