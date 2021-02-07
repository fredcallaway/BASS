using BenchmarkTools
# using Profile, ProfileView

include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")


function bench_sim()
    m = BDDM(cost=1e-4, risk_aversion=16e-3)
    pol = DirectedCognition(m)
    trials = prepare_trials(all_data; dt=.01)

    simulate(m, pol, trials[1])
    @time mapreduce(+, trials) do t
        simulate(m, pol, t).rt
    end
end

bench_sim()

# %% --------

function run_sims()
    m = BDDM(cost=1e-4, risk_aversion=16e-3)
    pol = DirectedCognition(m)
    trials = prepare_trials(all_data; dt=.01)
    t = trials[1]
    for i in 1:10000
        sim = simulate(m, pol, t)
        (sim.choice, sim.rt, sim.timeout)
    end
    nothing
end


run_sims()
Profile.init(1000000, 1e-5)
@profview run_sims()

