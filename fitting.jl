include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")

using SpecialFunctions: digamma
using ProgressMeter
# %% --------

trials = prepare_trials(all_data; dt=.01)
max_rt = quantile([t.rt for t in trials], .99)
trials = filter(t-> t.rt ≤ max_rt, trials)
# %% --------

function is_hit((choice, rt), t, tol)
    t.choice == choice && abs(rt - t.rt) ≤ tol
end

function sample_choice_rt(m::BDDM, t::Trial, ε::Float64)
    if rand() < ε
        choice = rand(1:2)
        rt = rand(1:max_rt)
        (choice, rt)
    else
        sim = simulate(m, DirectedCognition(m), t)
        sim.timeout && return (-1, -1)
        (sim.choice, sim.rt)
    end
end

function fixed_loglike(m::BDDM, t::Trial; ε=.01, tol=0, N=10000)
    hits = 0
    for i in 1:N
        if is_hit(sample_choice_rt(m, t, ε), t, tol)
            hits +=1
        end
    end
    log((hits + 1) / (N + 1))
end

function ibs_loglike(m::BDDM, t::Trial; ε=.01, tol=0)
    k = 0
    while true
        k += 1
        if is_hit(sample_choice_rt(m, t, ε), t, tol)
            break
        end
        k == 1e6 && @warn "k = 1e6"
        k == 1e7 && @warn "k = 1e7"
        k == 1e8 && @warn "k = 1e8"
    end
    digamma(1) - digamma(k)  # below Eq 14
end

# %% --------
m = BDDM(cost=1e-4, risk_aversion=16e-3)
subj = first(unique(t.subject for t in trials))
subj_trials = filter(t->t.subject == subj, trials)
results1 = @showprogress map(trials) do t
    @timed ibs_loglike(m, t, tol=10, ε=.01)
end;

results2 = @showprogress map(trials) do t
    @timed fixed_loglike(m, t, tol=10, ε=.01)
end;


