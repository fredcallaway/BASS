using SplitApplyCombine

include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("figure.jl")
using Printf
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
        sim = simulate(m, pol; save_states=true, max_rt=200)
        plot_sim(sim)
    end
    plot(plots...)
end

# %% ==================== likelihood ====================

trials = prepare_trials(all_data; dt=.01);
subj = first(unique(t.subject for t in trials))
subj_trials = filter(t->t.subject == subj, trials);

m = BDDM(cost=2.3e-4, risk_aversion=6e-2, base_precision=.01, attention_factor=.8)

i = findfirst(trials) do t
    t.value[1] == t.value[2] && t.value[1] < -1
end
i = 9
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

