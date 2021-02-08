# include("utils.jl")
# include("model.jl")
# include("dc.jl")
# include("data.jl")

const MAX_RT = 200

function is_hit((choice, rt), t, tol)
    t.choice == choice && abs(rt - t.rt) ≤ tol
end

function sample_choice_rt(m::BDDM, t::Trial, ε::Float64)
    if rand() < ε
        choice = rand(1:2)
        rt = rand(1:MAX_RT)
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

function chance_loglike(trials; tol)
    mapreduce(+, trials) do t
        n_within_tol = 1 + min(MAX_RT, t.rt + tol) - max(1, t.rt - tol)
        log(0.5) + log(n_within_tol / MAX_RT)
    end
end

function ibs_loglike(m::BDDM, trials::Vector{HumanTrial}; repeats=5, ε=.01, tol=0, min_multiplier=1)
    min_logp = min_multiplier * chance_loglike(trials; tol)
    ibs(trials; repeats, min_logp) do t
        is_hit(sample_choice_rt(m, t, ε), t, tol)
    end
end

