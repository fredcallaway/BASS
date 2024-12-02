# %% --------

data <- read_csv("/tmp/data_longpres.csv") |> 
    filter(val1 != val2) |> 
    mutate(correct = (choice == 1) == (val1 > val2))

# %% --------

data |> 
    group_by(attention_factor) |> 
    summarize(mean(correct))

data |> 
    group_by(attention_factor) |> 
    summarize(mean(is.na(initpresdur2)))

data |> 
    drop_na(initpresdur1, initpresdur2) |> 
    ggplot(aes(val1 + val2, 1 * (choice == 1), color=factor(initpresdur1 > initpresdur2))) +
    # stat_summary_bin(fun.data=mean_cl_normal, bins=10, geom="pointrange") +
    gam_fit() +
    facet_grid(~attention_factor)
fig("attention_bias_shortpres",w=7)

# %% --------

data |> 
    ggplot(aes(val1 - val2, 1 * (choice == 1))) +
    gam_fit()
fig(w=7)

# %% --------
data |> 
    mutate(correct=(choice == 1) == (val1 > val2)) |> 
    ggplot(aes(attention_factor, 1*correct)) +
    points()
fig(w=7)
# %% --------

source("base.r")

# version <- "recovery/2024-11-20"
# version <- "recovery-artificial/2024-11-23"
version <- "recovery-artificial/2024-11-25B"

generating <- read_csv(glue("results/{version}/generating_params.csv")) |> 
    select(!starts_with("prior"))

likelihoods <- read_csv(glue("results/{version}/likelihoods.csv"))

mle <- likelihoods |> 
    group_by(param_id, cost, base_precision, attention_factor) |> 
    summarize(
        logp = sum(logp),
        sd = sqrt(sum(std ^ 2))
    ) |> 
    group_by(param_id) |> 
    slice_max(logp) |> 
    left_join(generating, by="param_id", suffix=c("_fit", "_true"))

# %% --------
mle |>
    select(-c(logp, sd)) |> 
    pivot_longer(
        cols = -param_id,
        names_pattern = "(.+)_(true|fit)",
        names_to = c("param", "type"),
        values_to = "value"
    ) |> 
    pivot_wider(
        names_from = type,
        values_from = value
    ) |> 
    ggplot(aes(true, fit)) +
    geom_abline(slope=1, intercept=0) +
    geom_point(alpha=0.2) +
    stat_summary(fun=mean, geom="point", shape='*', size=10, color=RED) +
    facet_wrap(~param, scales="free") +
    gam_fit(k=3, color=RED, fill=RED, alpha=0.1) +
    gridlines +
    theme(aspect.ratio = 1)

fig("recovery_artificial",w=7)
