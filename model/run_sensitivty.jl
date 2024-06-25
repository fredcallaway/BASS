@everywhere include("sensitivity.jl")
# %% --------



data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

run_sensitivity("1-main", data1, Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1 / σ^2,
))

run_sensitivity("1-biased_mean", data1, Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = 0.8µ,
    prior_precision = 1 / σ^2,
))

run_sensitivity("1-zero_mean", data1, Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = 0,
    prior_precision = 1 / σ^2,
))

run_sensitivity("1-flat_prior", data1, Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1e-8,
))

# %% ==================== study 2 ====================



data2 = load_human_data(2)

µ, σ = empirical_prior(data2)
m2_main = BDDM(
    base_precision = 0.0005,
    confidence_slope = .008,
    attention_factor = 0.8,
    cost = .06,
    prior_mean = µ,
    prior_precision = 1 / σ^2
)