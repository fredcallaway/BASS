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


# %% ==================== Default parameters ====================


trials = prepare_trials(Table(data); dt=.025, normalize_value=false)
m = ADDM(0.3, 0.0002 * 25, .02 * 5)
sim_trials = map(trials) do t
    while true
        sim = simulate(m, SimTrial(t), save_fixations=true, max_t=max_rt(t))
        if sim.choice != -1
            return mutate(t; sim.choice, sim.rt, real_presentation_times=sim.fix_times)
        end
    end
end


# %% ==================== Grid ====================

box = Box(
    θ = (0, 1),
    d = (.001, .1, :log),
    σ = (10^-2, 10^-0.5, :log),
)

candidates = map(grid(10, box)) do g
    ADDM(;g...)
end

subjects = rand(eachindex(candidates), 100)
data = group(d->d.subject, all_data) |> first
trials = prepare_trials(Table(data); dt=.025, normalize_value=false)

# %% --------

sim_trials = map(subjects) do subject
    m = candidates[subject]
    sim = map(trials) do t
        sim = simulate(m, SimTrial(t), save_fixations=true, max_t=max_rt(t))
        return mutate(t; sim.choice, sim.rt, real_presentation_times=sim.fix_times, subject)
    end
    filter!(t->t.choice != -1, sim)
end;

okay = length.(sim_trials) .≥ 220
subjects = subjects[okay];
sim_trials = sim_trials[okay];

# %% --------
run_name = "addm/recovery2"
mkpath("tmp/$run_name")
map(sim_trials) do trials
    subj = trials[1].subject
    @assert all(t.subject == subj for t in trials)
    out = "tmp/$run_name/$subj"
    if isfile(out)
        println("$out already exists")
        return
    end
    println("Fitting subject $subj")
    ibs_kws = (ε=.01, tol=1, repeats=10, min_multiplier=1.2)
    results = @showprogress pmap(candidates) do m
        ibs_loglike(m, trials[1:2:end]; ibs_kws...)
    end
    chance = chance_loglike(trials[1:2:end]; ibs_kws.tol)
    serialize(out, (;box, trials, results, chance, ibs_kws, true_model=m))
    return results
end

map(sim_trials) do trials
    (table(trials).rt |> mean) / .025
end


