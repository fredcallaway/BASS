m = BDDM(cost=.05)
pol = CantStopWontStop()

function get_analytic(t, s, max_step)
    λ_avg = average_precision(m, t)
    tr = term_reward(m, s)
    map(0:max_step) do n
        n == 0 && return tr
        voc_n(m, s, n, λ_avg, t.dt) + tr
    end
end

"Samples a Trial with fixed confidence and values conditional on the DDM state"
function resample_values(t::Trial, s::State)
    value = @. s.μ + (s.λ ^ -0.5) * $randn(2)
    mutate(t, value=value)
end

function get_empirical(t, s, max_step; N=10000)
    tr = N \ mapreduce(+, 1:N) do i
        sim = simulate(m, resample_values(t, s); pol, s=deepcopy(s), max_step, save_states=true)
        map(sim.states) do ss
            term_reward(m, ss)
        end
    end
    cost = (0:max_step) .* m.cost .* t.dt
    tr .- cost
end

# %% ==================== With alternating samples ====================

function plot_comparison!(t, s)
    plot!(get_empirical(t, s, 200); color=:black)
    plot!(get_analytic(t, s, 200); color=1)
    term_reward(m, s)
end

figure() do
    presentation_distributions = [Normal(0.025, 1e-10), Normal(0.025, 1e-10)]
    t = SimTrial(;presentation_distributions, dt=.1)
    s = State(m)
    plot_comparison!(t, s)
    plot!(xlabel="Time steps", ylabel="Reward")
end

figure() do 
    ps = map(1:9) do i
        t = SimTrial()
        steps_per_cycle = Int(round(sum(mean.(presentation_distributions)) / t.dt, digits=4))
        s = simulate(m, pol; t, max_step=steps_per_cycle).states[1]    
        plot()
        plot_comparison!(t, s)
    end
    plot(ps..., size=(600,500))
end


# %% ==================== With random presentation duration ====================

figure() do
    t = SimTrial()
    s = State(m)
    plot_comparison!(t, s)
    plot!(xlabel="Time steps", ylabel="Reward")
end

figure() do 
    ps = map(1:9) do i
        t = SimTrial()
        s = simulate(m, pol; t, max_step=28).states[1]    
        plot()
        plot_comparison!(t, s)
    end
    plot(ps..., size=(600,500))
end

