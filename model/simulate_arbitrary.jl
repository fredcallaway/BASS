include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("box.jl")
using Serialization

# %% --------


function make_frame(data)
   map(data) do d
       val1, val2 = d.value
       conf1, conf2 = d.confidence
       pt1 = round(sum(d.presentation_duration[1:2:end]); digits=3)
       pt2 = round(sum(d.presentation_duration[2:2:end]); digits=3)
       (;d.subject, val1, val2, conf1, conf2, pt1, pt2, d.choice, d.rt)
   end |> Table
end

function simulate_dataset(m, trials; ndt=0)
    map(trials) do t
        sim = simulate(m, SimTrial(t); save_presentation=true)
        presentation_duration = t.dt .* sim.presentation_durations
        m1, m2 = mean.(t.presentation_distributions)
        order = m1 > m2 ? :longfirst : :shortfirst
        rt = sim.time_step .* t.dt + ndt
        (;t.subject, t.value, t.confidence, presentation_duration, order, sim.choice, rt)
    end
end

function write_sim(model, study)
    data = load_human_data(study)
    trials = repeat(prepare_trials(Table(data); dt=.1), 10);
    val_μ, val_σ = juxt(mean, std)(flatten(data.value))

    df = make_frame(simulate_dataset(model, trials))
    @. df.val1 = round(df.val1 * val_σ + val_μ; digits=2)
    @. df.val2 = round(df.val2 * val_σ + val_μ; digits=2)
    fn = "results/hand_sim_$study.csv"
    df |> CSV.write(fn)
    println("wrote $fn")
    df
end

# %% ==================== Study 1 main ====================

#m = deserialize("tmp/v7-2-best")
m = BDDM(
    base_precision = 0.25,
    attention_factor = 0.8,
    cost = .02,
    prior_mean = -0.1,
    risk_aversion=.3  # TODO
)
write_sim(m, 2)

# %% ==================== Study 1 zero prior ====================

#m = deserialize("tmp/v7-2-best")
m = BDDM(
    base_precision = 0.25,
    attention_factor = 0.8,
    cost = .02,
    prior_mean = -0.1,
    risk_aversion=.3  # TODO
)
write_sim(m, 2)

# %% ==================== Study 2 main ====================

m = BDDM(
    base_precision = 0.01,
    confidence_slope = .06,
    attention_factor = 0.8,
    cost = .019,
    prior_mean = -0.5,
)

write_sim(m, 3)















# %% ==================== Scratch ====================

m = deserialize("tmp/v7-3-best")

m = mutate(m,
    base_precision = 0.005,
    confidence_slope = .07,
    attention_factor = 0.8,
    cost = .02,
    prior_mean = -0.5,
)

write_sim(m, 3)


# %% --------

data = load_human_data(3)
trials = repeat(prepare_trials(Table(data); dt=.1), 1);
# %% --------
m = deserialize("tmp/v7-3-best")

m = mutate(m,
    base_precision = 0.01,
    confidence_slope = .05,
    attention_factor = 0.8,
    cost = .02,
    prior_mean = 0,
)
println(mean(log.(1000 .* invert(simulate_dataset(m, trials)).rt)))


