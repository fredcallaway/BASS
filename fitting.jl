include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
# %% --------

trials = prepare_trials(all_data)


# %% ==================== Plots ====================
include("figure.jl"); start_watch()

# %% --------

m = BDDM(cost=1e-4, risk_aversion=16e-3)
pol = DirectedCognition2(m)
trials[1]
@time sims = map(1:10000) do i
    sim = simulate(m, pol, trials[1])
    (sim.choice, sim.rt, sim.timeout)
end

choice, rt, timeout = invert(sims)
S = Table((; choice, rt, timeout))

using Printf

data[1].value


figure() do
    rt = sum(t.real_presentation_times)
    # plot(ylim=(0, 300))
    vline!(cumsum(t.real_presentation_times), color=:gray, alpha=0.4)
    title!(@sprintf("%.2f vs. %.2f", t.value...))
    histogram!(filter(x->x.choice == 2, S).rt; bins=0:rt, lw=0, alpha=0.5, color="#E54545", label="choose second")
    histogram!(filter(x->x.choice == 1, S).rt; bins=0:rt, lw=0, alpha=0.5, color="#36B5FF", label="choose first")
end

# %% --------

function get_weird_sim()
    for i in 1:1000
        sim = simulate(m, pol, trials[1], save_states=true)
        if !sim.timeout && sim.choice == 2 && sim.rt == 42
            # println("found on attempt #$i")
            return sim
        end
    end
end

sims = [get_weird_sim() for i in 1:1000]
# %% --------
weird_final = map(sims) do t
    t.states[end]
end

# %% --------

normal42 = map(1:1000) do i
    for j in 1:1000
        sim = simulate(m, pol, trials[1]; max_rt=42)
        if sim.timeout
            return sim.states[end]
        end
    end
end
# %% --------
plot_μ!(states) = scatter!(invert([s.μ for s in states])...)

figure() do
    plot()
    plot_μ!(normal42)
    plot_μ!(weird_final)
end



# %% --------
# figure() do
#     map(sim.states) do s
#         voc_dc(m, s, t)
#     end |> plot
# end

figure() do
    plot_sim(sim)
end
