include("base.jl")

version = "oct15"
mkpath("results/$version")


# %% ==================== Study 1 main ====================

#Rating Study 1: mean 4.789534
#SD 3.007145

data1 = load_human_data(2)
μ, σ = empirical_prior(data1, α=0.8)

m1_main = BDDM(
    base_precision = .05,
    attention_factor = 0.8,
    cost = .06,
    prior_mean = μ,
    prior_precision = 1 / σ^2,
)

df = write_sim(m1_main, data1, "1-main")

# %% ==================== Study 1 flat prior ====================

#m = deserialize("tmp/v7-2-best")
m1_flat = mutate(m1_main,
    prior_precision = 1e-6
)
df = write_sim(m1_flat, data1, "1-flatprior")

# %% ==================== Study 1 zero prior ====================

#m = deserialize("tmp/v7-2-best")
m1_zero = mutate(m1_main,
    prior_mean = 0
)
write_sim(m1_zero, data1, "1-zeroprior")

# %% ==================== Study 1 unbaised prior ====================

#m = deserialize("tmp/v7-2-best")
m1_unbiased = mutate(m1_main,
    prior_mean = empirical_prior(data1, α=1.)[1]
)
write_sim(m1_unbiased, data1, "1-unbiasedprior")

# %% ==================== Study 2 main ====================
#Study2: 5.134873
#3.069003

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

# %% ==================== Study 2 no metacognition ====================

avg_confidence = m2_main.confidence_slope * mean(flatten(data2.confidence))
m2_nometa = mutate(m2_main,
    subjective_slope = 0,
    subjective_offset = avg_confidence
)
df = write_sim(m2_nometa, data2, "2-nometa")

# %% ==================== Study 2 null confidence ====================
m2_null = mutate(m2_main,
    confidence_slope = 0,
    base_precision = m2_main.base_precision + avg_confidence
)
df = write_sim(m2_null, data2, "2-ignoreconf")

# %% ==================== Study 2 overconfidence ====================

function make_sim(models::Vector{BDDM}, data; kws...)
    mapreduce(vcat, models) do model
        df = make_sim(model, data; kws...)
        Table(df; 
            subjective_offset = fill(model.subjective_offset, length(df)),
            subjective_slope = fill(model.subjective_slope, length(df))
        )
    end
end

over_models = map(0:.005:.04) do subjective_offset
    mutate(m2_main; subjective_offset)
end

df = write_sim(over_models, data2, "2-overconf"; repeats=5)

# %% ==================== Study 2 overconfidence-slope ====================

slope_over_models = map(.6:.2:1.4) do subjective_slope
    mutate(m2_main; subjective_slope)
end

df = write_sim(slope_over_models, data2, "2-overconfslope"; repeats=5)

# %% ==================== Study 2 no bias ====================

m2_nobias = mutate(m2_main,
    prior_mean = empirical_prior(data2)[1]
    #confidence_slope = 0,
    #base_precision = m2_main.base_precision + avg_confidence
)
df = write_sim(m2_nobias, data2, "2-nobias")

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


