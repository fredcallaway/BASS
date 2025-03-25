include("base.jl")

version = "mar25-fit"
mkpath("results/$version")


# %% ==================== Study 1 main ====================

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)
m1_main = BDDM(
    base_precision = 0.05,
    attention_factor = 0.0,
    cost = 0.05,
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)

df = write_sim(m1_main, data1, version, "1-main")

# %% ==================== Study 1 flat prior ====================

m1_flat = mutate(m1_main,
    prior_precision = 1e-6
)
df = write_sim(m1_flat, data1, version, "1-flatprior")

# %% ==================== Study 1 zero prior ====================

m1_zero = mutate(m1_main,
    prior_mean = 0
)
write_sim(m1_zero, data1, version, "1-zeroprior")

# %% ==================== Study 2 main ====================

data2 = load_human_data(2)

µ, σ = empirical_prior(data2)
m2_main = BDDM(
    base_precision = 0.01,
    confidence_slope = 0.01,
    attention_factor = 0.0,
    cost = .05,
    prior_mean = µ,
    prior_precision = 1 / σ^2
)
df = write_sim(m2_main, data2, version, "2-main")

# %% ==================== Study 2 no metacognition ====================

avg_confidence = m2_main.confidence_slope * mean(flatten(data2.confidence))
m2_nometa = mutate(m2_main,
    subjective_slope = 0,
    subjective_offset = avg_confidence
)
df = write_sim(m2_nometa, data2, version, "2-nometa")

# %% ==================== Study 2 overconfidence ====================

over_models = map(0:.005:.04) do subjective_offset
    mutate(m2_main; subjective_offset)
end

df = write_sim(over_models, data2, version, "2-overconf"; repeats=5)


#= ... unused, left in for posterity ...

# %% ==================== Study 1 biased prior ====================

#m = deserialize("tmp/v7-2-best")
m1_biased = mutate(m1_main,
    prior_mean = empirical_prior(data1, α=0.7)[1]
)
write_sim(m1_biased, data1, version, "1-biased")

# %% ==================== Study 2 biased prior ====================

m2_biased = mutate(m2_main,
    prior_mean = empirical_prior(data2, α=0.7)[1]
)
write_sim(m2_biased, data1, version, "2-biased")

# %% ==================== Study 2 overconfidence-slope ====================

slope_over_models = map(.6:.2:1.4) do subjective_slope
    mutate(m2_main; subjective_slope)
end

df = write_sim(slope_over_models, data2, "2-overconfslope"; repeats=5)

=#