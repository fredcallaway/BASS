using Distributed
@everywhere begin
    include("base.jl")
    include("likelihood.jl")
    include("grid_search.jl")
    using Serialization
end
using SplitApplyCombine


experiment = ARGS[1]
job = ARGS[2]
version = "2025-04-07-thetalarge-$experiment"

if job == "process"
    # version = "2024-11-04"
    using DataFrames, CSV
    mkpath("results/grid")
    df = flatmap(readdir("tmp/bddm/grid/$version")) do subject
        x = deserialize("tmp/bddm/grid/$version/$subject")
        map(x.candidates[:], x.results[:]) do c, r
            if r isa Float64
                logp = r
                std = 0.0
                converged = false
            else
                logp = r.logp
                std = r.std
                converged = r.converged
            end
            (; subject, c.base_precision, c.attention_factor, c.confidence_slope, c.cost, logp, std, converged)
        end
    end |> DataFrame
    df |> CSV.write("results/grid/$version.csv", writeheader=true)
    prms = [:base_precision, :confidence_slope, :attention_factor, :cost]
    total = combine(groupby(df, prms), :logp => mean => :logp)
    println(sort(total, :logp, rev=true)[1:5, :])
else
    fit_mode = if job == "group"
        :group
    elseif job == "subjects"
        :subjects
    else
        parse(Int, job)
    end

    if experiment == "1"
        data = load_human_data(1)
        µ, σ = empirical_prior(data)
        box = Box(
            base_precision = (0.0, 0.1),
            attention_factor = (1, 2),
            cost = (0.0, 0.1),
            prior_mean = µ,
            prior_precision = 1 / σ^2,
        )
    elseif experiment == "2"
        data = load_human_data(2)
        µ, σ = empirical_prior(data)
        box = Box(
            base_precision = (0.0, .05),
            confidence_slope = (0.0, .02),
            attention_factor = (0.0, 1.0),
            cost = .06,
            prior_mean = µ,
            prior_precision = 1 / σ^2,
        )
    else
        error("Invalid experiment: $experiment")
    end

    n_grid = 11
    grid_search(BDDM, version, box, n_grid, data, repeats=100, ε=.05; tol=0, fit_mode)
end
