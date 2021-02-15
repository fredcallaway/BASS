@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
    using Serialization
end
using ProgressMeter
using Sobol
using SplitApplyCombine

# %% --------

box = Box(
    base_precision = (.01, 2, :log),
    attention_factor = (0, 2),
    cost = (.01, .1, :log),
    confidence_slope = (0, 1),
    prior_mean = (-1, 1),
)

n_subj = 100
xs = Iterators.take(SobolSeq(n_free(box)), n_subj) |> collect

# %% --------
trial_sets = map(collect(group(d->d.subject, all_data))) do data
    SimTrial.(prepare_trials(Table(data), dt=.1))
end |> Iterators.cycle;
# %% --------

@showprogress "Simulating" pmap(enumerate(xs), trial_sets) do (i, x), trials
    m = BDDM(;box(x)...)
    map(trials) do t
        sim = simulate(m, t)
        HumanTrial(t.value, t.confidence, t.presentation_times, subject, sim.choice, sim.time_step, t.dt)
    end
end




