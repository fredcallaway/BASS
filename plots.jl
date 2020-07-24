using SplitApplyCombine

include("model.jl")
include("figure.jl")
toggle_watch()

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
