include("base.jl")

version = "may7-sepfit"
mkpath("results/$version")

# %% ==================== Load data ====================

data1 = load_human_data(1)
data2 = load_human_data(2)
avg_conf = mean(flatten(data2.confidence))

# use average confidence from study 2 for study 1 (no confidence judgments)
for d in data1
    d.confidence .= avg_conf
end

make_frame(data1) |> CSV.write("data/study1.csv")
make_frame(data2) |> CSV.write("data/study2.csv")

# %% ==================== Define model parameters ====================

summary_fits = CSV.read("results/summary_fits.csv", DataFrame)

models = map(eachrow(summary_fits)) do row
    model = BDDM(;
        row.base_precision,
        row.confidence_slope,
        row.attention_factor,
        row.cost,
        row.prior_mean,
        row.prior_precision,
        row.subjective_offset,
        row.subjective_slope,
    )
    (row.study, row.model) => model
end |> Dict


# %% ==================== Study 1 ====================

write_sim(models[1, "main"], data1, version, "1-main")
write_sim(models[1, "flat_prior"], data1, version, "1-flatprior")
write_sim(models[1, "zero_mean"], data1, version, "1-zeroprior")

# %% ==================== Study 2 ====================

write_sim(models[2, "main"], data2, version, "2-main")
write_sim(models[2, "nometa"], data2, version, "2-nometa")

over_models = map(0:.005:.04) do subjective_offset
    mutate(models[2, "main"]; subjective_offset)
end
write_sim(over_models, data2, version, "2-overconf"; repeats=5)
