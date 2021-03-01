using SplitApplyCombine

include("utils.jl")
include("model.jl")
include("addm.jl")
include("dc.jl")
include("data.jl")
include("figure.jl")
include("likelihood.jl")
using Printf
using Query
# %% ==================== diffusion ====================

function plot_sim(sim)
    μs, λs = map(sim.states) do s
        s.μ, s.λ
    end |> invert .|> combinedims
    plot(μs', ribbon=λs' .^ -0.5; fillaplha=0.1,
         xaxis=("", []), yaxis=("", []), framestyle = :origin)
end

figure("diffusion") do
    m = BDDM()
    pol = CantStopWontStop()
    plots = map(1:9) do i
        t = SimTrial(dt=.1)
        sim = simulate(m, t; pol, save_states=true)
        plot_sim(sim)
    end
    plot(plots...)
end

# %% ====================  ====================

figure("diffusion") do
    m = BDDM(cost=.001, base_precision=.01)
    pol = DirectedCognition(m)
    plots = map(1:9) do i
        dt = .01
        t = SimTrial(;dt, value=[-1, 1])
        sim = simulate(m, t; pol, save_states=true)
        sim.time_step * dt
        μs, λs = map(sim.states) do s
            s.μ, s.λ
        end |> invert .|> combinedims

        plot(collect((0:sim.time_step)) .* dt, μs', ribbon=λs' .^ -0.5; fillalpha=0.2,
             xaxis=("", [], ), xlim=(0, 10), yaxis=("", [],), framestyle = :origin)
    end
    plot(plots...)
end

# %% ==================== likelihood ====================

subj = first(unique(t.subject for t in trials))
subj_trials = filter(t->t.subject == subj, trials);
sample_choice_rt(m, subj_trials[1], .05)
# m = BDDM()
# %% --------
m = deserialize("tmp/good_model")
m = mutate(m, base_precision=40m.base_precision, cost=m.cost)
average_precision(m, t)
# %% --------

i = findfirst(trials) do t
    t.value[1] == t.value[2] && t.value[1] < -1
end
i = 7
t = trials[i]

@time S = map(1:10000) do i
    choice, rt = sample_choice_rt(m, t, 0.)
    (; choice, rt)
end |> Table

figure() do
    rt = sum(t.real_presentation_times)
    # plot(ylim=(0, 300))
    vline!(cumsum(t.real_presentation_times), color=:gray, alpha=0.4)
    title!(@sprintf("%.2f vs. %.2f", t.value...))
    # histogram!(filter(x->x.choice == -1, S).rt; bins=-1:rt, lw=0, alpha=1, color="#FFDD47", label="timeout")
    histogram!(filter(x->x.choice == 2, S).rt; bins=-1:rt, lw=0, alpha=0.5, color="#E54545", label="choose second")
    histogram!(filter(x->x.choice == 1, S).rt; bins=-1:rt, lw=0, alpha=0.5, color="#36B5FF", label="choose first")
end

# %% ==================== ADDM ====================

data = group(d->d.subject, all_data) |> first

dt = .025
trials = prepare_trials(Table(data); dt, normalize_value=false)
m = ADDM(reference=5)

# trials = prepare_trials(Table(data); dt=.001, normalize_value=false)
# m = ADDM()

# ibs_loglike(m, trials[1:2:end]; ε=.01, tol=10, repeats=10, min_multiplier=1)
# ibs_loglike(m, trials[1:2:end]; ε=1., tol=10, repeats=100, min_multiplier=2)
# chance_loglike(trials[1:2:end]; tol=10)

t = trials |> @filter(_.value[1] < 5) |> collect |> rand

@time S = map(1:100000) do i
    select(simulate(m, t), (:choice, :rt))
end |> Table

figure() do
    # plot(ylim=(0, 300))
    vline!(cumsum(t.real_presentation_times), color=:gray, alpha=0.4)
    if t.choice == 1
        title!(@sprintf("[[%.2f]] vs. %.2f - %d%% overshot", t.value..., 100*mean(S.choice .== -1)))
    else
        title!(@sprintf("%.2f vs. [[%.2f]] - %d%% overshot", t.value..., 100*mean(S.choice .== -1)))
    end
    g = group(x->x.choice, S)
    bins = 1:ceil(Int, .025 / dt):t.rt

    # bins = 1:20:t.rt
    # histogram!(g[-1].rt; bins, lw=0, alpha=1, color="#FFDD47", label="timeout")
    histogram!(g[2].rt; bins, lw=0, alpha=0.5, color="#E54545", label="choose second")
    histogram!(g[1].rt; bins, lw=0, alpha=0.5, color="#36B5FF", label="choose first")
end

# %% --------
hist = simulate(m, t, save_history=true).history

figure() do
    plot!(xaxis=("", []), yaxis=("", [], (-1, 1)), framestyle = :origin)
    title!(@sprintf("%.2f vs. %.2f", t.value..., ))
    vline!(cumsum(t.real_presentation_times), color=:gray, alpha=0.4)
    plot!(hist)
end



