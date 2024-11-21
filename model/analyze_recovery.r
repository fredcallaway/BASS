# %% --------

source("base.r")

version <- "2024-11-20"

generating <- read_csv(glue("results/recovery/{version}/generating_params.csv")) |> 
    select(!starts_with("prior"))

likelihoods <- read_csv(glue("results/recovery/{version}/likelihoods.csv"))

mle <- likelihoods |> 
    group_by(param_id, cost, base_precision, attention_factor) |> 
    summarize(
        logp = sum(logp),
        sd = sqrt(sum(std ^ 2))
    ) |> 
    filter(!is.na(sd)) |> 
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
    facet_wrap(~param, scales="free") +
    gam_fit(k=3) +
    gridlines +
    theme(aspect.ratio = 1)

fig(w=7)
