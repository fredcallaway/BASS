include("ibs.jl")

function is_hit((choice, rt), t, tol)
    if tol == -1  # ignore rt
        return t.choice == choice
    else
        return t.choice == choice && t.rt > 0 && abs(rt - t.rt) ≤ tol
    end
end

function sample_choice_rt(m, t::Trial, ε)
    if rand() < ε
        choice = rand(1:2)
        rt = rand(1:max_rt(t))
        (choice, rt)
    else
        sim = simulate(m, t)
        sim.timeout && return (sim.choice, -1)  # -1 means timeout (excluded in is_hit)
        (sim.choice, sim.time_step)
    end
end

function fixed_loglike(m, t::Trial; ε=.05, tol=0, N=10000)
    hits = 0
    for i in 1:N
        if is_hit(sample_choice_rt(m, t, ε), t, tol)
            hits +=1
        end
    end
    log((hits + 1) / (N + 1))
end

function chance_loglike(trials; tol)
    tol == -1 && return length(trials) * log(0.5)
    mapreduce(+, trials) do t
        n_within_tol = 1 + min(max_rt(t), t.rt + tol) - max(1, t.rt - tol)
        log(0.5) + log(n_within_tol / max_rt(t))
    end
end

function ibs_loglike(m, trials::Vector{HumanTrial}; repeats=1, ε=.05, tol=0, min_multiplier=2.0)
    min_logp = min_multiplier * chance_loglike(trials; tol)
    ibs(trials; repeats, min_logp) do t
        is_hit(sample_choice_rt(m, t, ε), t, tol)
    end
end

