@everywhere begin
    using Parameters
    using Serialization
    using SplitApplyCombine
    using Statistics
    using Optim
    using Sobol

    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
    include("particle_filter.jl")
end

@everywhere begin
    data = filter(d->d.subject == 1064, all_data)
    trials = prepare_trials(data; dt=.025)
    m = deserialize("tmp/good_model")
end

# %% --------
simulate(m::BDDM, t::HumanTrial; kws...) = simulate(m, DirectedCognition(m), t; kws...)
chance_loglike(trials[1:2:end], tol=0)
@time ibs_loglike(m, trials[1:2:end]; ε=.05, tol=0, min_multiplier=1.2)

# %% --------
# simulate(m::BDDM, t::HumanTrial; kws...) = simulate(m, MetaGreedy(m, 20), t; kws...)
@time out = ibs_loglike(m, trials[1:100]; ε=0.1, tol=0, min_multiplier=1.2, repeats=10)
out.n_call / 100
ibs_loglike(m, trials[1:100]; ε=0.1, tol=0, min_multiplier=1.2, repeats=10)

# %% ==================== Test Particle ====================
history = []
t = trials[1]
pf = ParticleFilter(m, DirectedCognition(m), t)
lp = run!(pf, n_particle)
run!(pf) do


map(history[1]) do ps
    ps.splits
end |> sum





# %% ==================== Compare IBS and Particle Filter ====================

using ProgressMeter
@everywhere m = BDDM()

ibs_results = @showprogress pmap(1:500) do i
    @timed ibs_loglike(m, trials[1:100]; ε=0.1, tol=0, min_multiplier=1.2, repeats=1)
end
 # 68 seconds
# %% --------

@everywhere function pf_loglike(m, trials::Vector{HumanTrial}; n_particle=3000, ε=.1)
    mapreduce(+, trials) do t
        pf = ParticleFilter(m, DirectedCognition(m), t)
        lp = run!(pf, n_particle)
        chance_lp = log(0.5) + log(1 / max_rt(t))
        logaddexp(log(1 - ε) + lp, log(ε) + chance_lp)
    end
end

pf_results = @showprogress pmap(1:500) do i
    @timed pf_loglike(m, trials[1:100]; n_particle=1000, ε=0.1)
end

chance_loglike(trials[1:100]; tol=0)

# %% --------
ibs_logp, ibs_time = map(ibs_results) do r
    r.value.logp, r.time
end |> invert

pf_logp, pf_time = map(pf_results) do r
    r.value, r.time
end |> invert

@show mean(pf_time)
@show mean(ibs_time)

td = mean(pf_time) / mean(ibs_time)
@show std(ibs_logp)
@show std(ibs_logp) / √td
@show std(pf_logp)

@show mean(pf_logp)
@show mean(ibs_logp)


# %% --------
ibs_logp = getfield.(ibs_results, :logp)
mean(pf_results)
mean(ibs_logp)

std(ibs_logp)
std(pf_results)



