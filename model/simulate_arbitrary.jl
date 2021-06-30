include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("box.jl")
using Serialization

# %% --------
version = "jun30"
mkpath("results/$version")

function empirical_prior(data; α=1)
    μ, σ = juxt(mean, std)(flatten(data.value))
    α * μ, σ
end

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

function write_sim(model, data, name; normalize_value=false)
    trials = repeat(prepare_trials(Table(data); dt=.1, normalize_value), 5);
    df = make_frame(simulate_dataset(model, trials))
    if normalize_value
        # unnomrmalize it
        val_μ, val_σ = empirical_prior(data)
        @. df.val1 = round(df.val1 * val_σ + val_μ; digits=2)
        @. df.val2 = round(df.val2 * val_σ + val_μ; digits=2)
    end
    fn = "results/$version/$name.csv"
    df |> CSV.write(fn)
    println("wrote $fn")
    df
end

# %% ==================== Study 1 main ====================

data1 = load_human_data(2)
μ, σ = empirical_prior(data1, α=0.8)

m1_main = BDDM(
    base_precision = .05,
    attention_factor = 0.8,
    cost = .05,
    prior_mean = μ,
    prior_precision = 1 / σ^2,
    #prior_precision = 1 / σ^2,
)
write_sim(m1_main, data1, "1-main")

#m = BDDM(
#    base_precision = 0.25 / σ^2,
#    attention_factor = 0.8,
#    cost = .02 * σ,
#    prior_mean = μ,
#    prior_precision = 1 / σ^2,
#)


# %% ==================== Study 1 flat prior ====================

#m = deserialize("tmp/v7-2-best")
m1_flat = mutate(m1_main,
    prior_precision = 1e-3
)
write_sim(m1_flat, data1, "1-flatprior")

# %% ==================== Study 1 zero prior ====================

#m = deserialize("tmp/v7-2-best")
m1_zero = mutate(m1_main,
    prior_mean = 0
)
write_sim(m1_zero, data1, "1-zeroprior")

# %% ==================== Study 2 main ====================

data2 = load_human_data(3)
μ, σ = empirical_prior(data2, α=0.7)
m2_main = BDDM(
    base_precision = 0.0005,
    confidence_slope = .008,
    attention_factor = 0.8,
    cost = .06,
    prior_mean = μ,
    prior_precision = 1 / σ^2
)

df = write_sim(m2_main, data2, "2-main")
println(mean(log.(1000 * df.rt)))

# %% ==================== Study 2 dumb confidence ====================

avg_confidence = m2_main.confidence_slope * mean(flatten(data2.confidence))
m2_dumb = mutate(m2_main,
    subjective_slope = 0,
    subjective_offset = avg_confidence
)
df = write_sim(m2_dumb, data2, "2-dumbconf")

# %% ==================== Study 2 average confidence ====================

avg_confidence = m2_main.confidence_slope * mean(flatten(data2.confidence))
m2_dumb = mutate(m2_main,
    confidence_slope = 0,
    base_precision = m2_main.base_precision + m2_main.confidence_slope * mean(flatten(data2.confidence))
)
df = write_sim(m2_dumb, data2, "2-averageconf")

# %% ==================== Scratch ====================

data = load_human_data(2)
trials = repeat(prepare_trials(Table(data); dt=.1), 10);
val_μ, val_σ = juxt(mean, std)(flatten(data.value))



# %% --------

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


