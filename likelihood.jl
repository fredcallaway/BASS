# include("utils.jl")
# include("model.jl")
# include("dc.jl")
# include("data.jl")

MAX_RT = 5
max_rt(t::Trial) = Int(MAX_RT / t.dt)

function is_hit((choice, rt), t, tol)
    t.choice == choice && abs(rt - t.rt) ≤ tol
end

function sample_choice_rt(m, t::Trial, ε)
    if rand() < ε
        choice = rand(1:2)
        rt = rand(1:max_rt(t))
        (choice, rt)
    else
        sim = simulate(m, t)
        sim.timeout && return (-1, -1)
        (sim.choice, sim.rt)
    end
end

function fixed_loglike(m, t::Trial; ε=.01, tol=0, N=10000)
    hits = 0
    for i in 1:N
        if is_hit(sample_choice_rt(m, t, ε), t, tol)
            hits +=1
        end
    end
    log((hits + 1) / (N + 1))
end

function chance_loglike(trials; tol)
    mapreduce(+, trials) do t
        n_within_tol = 1 + min(max_rt(t), t.rt + tol) - max(1, t.rt - tol)
        log(0.5) + log(n_within_tol / max_rt(t))
    end
end

function ibs_loglike(m, trials::Vector{HumanTrial}; repeats=1, ε=.01, tol=0, min_multiplier=1.2)
    min_logp = min_multiplier * chance_loglike(trials; tol)
    ibs(trials; repeats, min_logp) do t
        is_hit(sample_choice_rt(m, t, ε), t, tol)
    end
end

