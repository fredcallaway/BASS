@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
    using Serialization
end
using ProgressMeter
using Sobol
using SplitApplyCombine
include("run_sobol.jl")
mkpath("tmp/recovery")

# %% ==================== Simulate ====================

sim_box = Box(
    base_precision = (.01, 1),
    attention_factor = (0, 1),
    cost = (.01, .1),
    confidence_slope = (0, 0.25),
    prior_mean = (-1, 1),
)

n_subj = 100
xs = Iterators.take(SobolSeq(n_free(sim_box)), n_subj) |> collect

all_data = load_human_data()
trial_sets = map(collect(group(d->d.subject, all_data))) do data
    SimTrial.(prepare_trials(Table(data), dt=.001))
end |> Iterators.cycle  |> (x -> Iterators.take(x, length(xs)))

@everywhere function simulate_dataset(m, trials; subject=0)
    map(trials) do t
        sim = simulate(m, t)
        presentation_duration = t.dt .* sim.presentation_times
        m1, m2 = mean.(t.presentation_distributions)
        order = m1 > m2 ? :longfirst : :shortfirst
        rt = sim.time_step .* t.dt
        (;subject, t.value, t.confidence, presentation_duration, order, sim.choice, rt)
    end
end

sim_data = @showprogress "Simulating " map(enumerate(xs), trial_sets) do (subject, x), trials
    m = BDDM(;sim_box(x)...)
    simulate_dataset(m, trials; subject)
end

serialize("tmp/recovery/sim_data", sim_data)

# %% ==================== Fit ====================

run_sobol_ind(BDDM, "recovery/v1", box, 5000, repeats=10, dt=.025, tol=0; data=flatten(sim_data))



# %% ====================  ====================


