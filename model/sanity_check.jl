include("base.jl")

# %% --------

data1 = load_human_data(1)
µ, σ = empirical_prior(data1)

m1_main = BDDM(
    base_precision=0.05,
    attention_factor=0.8,
    cost=0.06,
    prior_mean=µ,
    prior_precision=1 / σ^2,
)

df = mapreduce(vcat, 0:.25:1.5) do attention_factor
    d = DataFrame(make_sim(mutate(m1_main; attention_factor), data1; repeats=10))
    d.attention_factor .= attention_factor
    d
end

CSV.write("results/sanity_check_empirical.csv", df)

# %% --------


# %% --------
Table(sim_data).subject |> countmap
countmap(data1.subject)

include("likelihood.jl")

grid_search(BDDM, "recovery-artifical/2024-11-21/1", box, 7, sim_data; repeats=10, ε=.05, tol=4)



trials = repeatedly() do _
    v1 = rand(Normal(µ, σ))
    v2 = rand(Normal(µ, σ))

    Trial(;value=[v1, v2], confidence=[1, 1], presentation_distributions=[Normal(0.2, 0.05), Normal(0.5, 0.1)])
end




df = mapreduce(vcat, 0:0.25:1.5) do attention_factor
    d = DataFrame(make_sim(mutate(m1_main; attention_factor), data1; repeats=10))
    d.attention_factor .= attention_factor
    d
end

CSV.write("results/sanity_check_empirical.csv", df)