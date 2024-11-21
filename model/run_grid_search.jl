using Distributed
@everywhere begin
    include("base.jl")
    include("likelihood.jl")
    using Serialization
end
using ProgressMeter
using SplitApplyCombine

# %% --------

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

box = Box(
    base_precision = (.01, .1),
    attention_factor = (0.5, 1.5),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)

version = "2024-10-24"
version = "2024-11-04"
grid_search(BDDM, version, box, 7, data1, repeats=10, ε=.05; tol=4)

# %% --------

# version = "2024-11-04"
using DataFrames, CSV
mkpath("results/grid")
flatmap(readdir("tmp/bddm/grid/$version")) do subject
    x = deserialize("tmp/bddm/grid/$version/$subject")
    map(x.candidates[:], x.results[:]) do c, r
        (; subject, c.base_precision, c.attention_factor, c.cost, r.logp, r.std, r.converged)
    end
end |> CSV.write("results/grid/$version.csv", writeheader=true)
