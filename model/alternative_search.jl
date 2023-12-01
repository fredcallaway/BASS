include("simulate_base.jl")

version = "oct15"
mkpath("results/$version")

# %% --------


data1 = load_human_data(1)
µ, σ = empirical_prior(data1, α=0.8)

# %% --------

box = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)

prms = grid(2, box)


# %% --------


m1_main = BDDM(
    base_precision = .05,
    attention_factor = 0.8,
    cost = .06,
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)


df = write_sim(m1_flat, data1, "1-flatprior")