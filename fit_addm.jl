@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("addm.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    using Serialization
end
using ProgressMeter
using Sobol
using SplitApplyCombine
include("box.jl")

# %% --------
m0 = ADDM()
box = Box(
    θ = (0, 1),
    d = (m0.d / 3, m0.d * 3, :log),
    σ = (m0.σ / 3, m0.σ * 3, :log),
)

# # trials = prepare_trials(Table(data); dt=.025, normalize_value=false)
# box = Box(
#     θ = (0, 1),
#     d = (.001, .1, :log),
#     σ = (10^-2, 10^-0.5, :log),
# )

candidates = map(grid(13, box)) do g
    ADDM(;g...)
end

run_name = "addm/grid/v7"
mkpath("tmp/$run_name")
# to_fit = [first(pairs(group(d->d.subject, all_data)))]
to_fit = pairs(group(d->d.subject, all_data))

results = map(to_fit) do (subj, data)
    out = "tmp/$run_name/$subj"
    if isfile(out)
        println("$out already exists")
        return
    end
    println("Fitting subject $subj")
    trials = prepare_trials(Table(data); dt=.025)
    filter!(trials) do t
        # this can happen due to rounding error
        t.rt <= max_rt(t)
    end
    # trials = prepare_trials(Table(data); dt=.001)
    ibs_kws = (ε=.01, tol=1, repeats=10, min_multiplier=1.2)
    results = @showprogress pmap(candidates) do m
        ibs_loglike(m, trials[1:2:end]; ibs_kws...)
    end
    chance = chance_loglike(trials[1:2:end]; ibs_kws.tol)
    serialize(out, (;box, trials, results, chance, ibs_kws))
    return results
end;

