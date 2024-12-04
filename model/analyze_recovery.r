# %% --------

source("base.r")

version <- "recovery/2024-11-20"
# version <- "recovery-artificial/2024-11-25B"
# version <- "recovery-artificial/2024-11-23"
# version <- "recovery-artificial-rapid/2024-11-25"

generating <- read_csv(glue("results/{version}/generating_params.csv")) |> 
    select(!starts_with("prior"))

likelihoods <- read_csv(glue("results/{version}/likelihoods.csv"))

total_likelihoods <- likelihoods |> 
    group_by(param_id, cost, base_precision, attention_factor) |> 
    filter(n() == max(n())) |> 
    summarize(
        logp = sum(logp),
        sd = sqrt(sum(std ^ 2))
    )

total_likelihoods |> filter(param_id == 27) |> 
    drop_na(sd) |> 
    with(sum(logp))

likelihoods |> 
    group_by(param_id, cost, base_precision, attention_factor) |> 
    filter(n() == max(n())) |> 
    ungroup() |> 
    count(param_id) |> 
    filter(param_id == 27)


# generating |> 
#     filter(between(attention_factor, .6, .7), base_precision == .07, cost==.04) |> 
    

mle <- total_likelihoods |> 
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
    mutate(param = case_when(
        param == "attention_factor" ~ "Attention Factor (θ)",
        param == "cost" ~ "Sampling Cost (c)",
        param == "base_precision" ~ "Baseline Precision (τ)"
    )) |> 
    ggplot(aes(true, fit)) +
    geom_abline(slope=1, intercept=0) +
    geom_point(alpha=0.2) +
    stat_summary(fun=mean, geom="point", shape='*', size=10, color=RED) +
    facet_wrap(~param, scales="free") +
    gam_fit(k=3, color=RED, fill=RED, alpha=0.1) +
    gridlines +
    theme(aspect.ratio = 1) +
    labs(x="True Parameter Value", y="MLE Parameter Value")

fig("recovery",w=7)

# %% --------

total_likelihoods |> 
    group_by(param_id, attention_factor) |> 
    slice_max(logp) |> 
    left_join(generating, by=c("param_id"), suffix=c("_fit", "_true")) |> 
    filter(cost_true == .04, base_precision_true == .04) |> 
    ggplot(aes(attention_factor_fit, logp)) +
    geom_line() +
    scale_color_brewer(palette="Set1") +
    geom_vline(aes(xintercept=attention_factor_true), linetype="dashed") +
    facet_wrap(~attention_factor_true, scales="free", nrow=2) +
    theme(strip.text.x = element_blank())
    # gam_fit(k=3, color=RED, fill=RED, alpha=0.1) +
    # gridlines +
    # theme(aspect.ratio = 1)

fig("recovery-attention",w=10, h=5)

# %% --------


total_likelihoods |> 
    group_by(param_id) |> 
    left_join(generating, by="param_id", suffix=c("_fit", "_true")) |>
    filter(cost_true == 0.04) |> 
    group_by(param_id, base_precision_fit, attention_factor_fit) |> 
    slice_max(logp) |> 
    group_by(param_id) |> 
    mutate(logp = logp - max(logp)) |> 
    ggplot(aes(attention_factor_fit, base_precision_fit, fill=logp)) +
    geom_tile() +
    geom_point(aes(x=attention_factor_true, y=base_precision_true), 
               shape='+', size=3, color='red') +
    scale_fill_viridis_c() +
    facet_grid(fct_rev(factor(base_precision_true)) ~ attention_factor_true, scales="free") +
    labs(x="Fitted Attention Factor (θ)", 
         y="Fitted Baseline Precision (τ)",
         fill="Log Likelihood") +
    theme(aspect.ratio = 1, 
          strip.text.x = element_blank(), 
          strip.text.y = element_blank())

fig("recovery-heatmap", w=10, h=5)

# %% --------
total_likelihoods |> 
    filter(param_id == 2) |> 
    group_by(param_id) |> 
    left_join(generating, by="param_id", suffix=c("_fit", "_true")) |>
    group_by(param_id, abse_precision_fit, attention_factor_fit) |> 
    slice_max(logp) |> 
    group_by(param_id) |> 
    mutate(logp = logp - max(logp)) |> 
    ggplot(aes(attention_factor_fit, base_precision_fit, fill=logp)) +
    geom_tile() +
    geom_point(aes(x=attention_factor_true, y=base_precision_true), 
               shape='+', size=3, color='red') +
    scale_fill_viridis_c() +
    facet_grid(fct_rev(factor(base_precision_true)) ~ attention_factor_true, scales="free") +
    labs(x="Fitted Attention Factor (θ)", 
         y="Fitted Baseline Precision (τ)",
         fill="Log Likelihood") +
    theme(aspect.ratio = 1, 
          strip.text.x = element_blank(), 
          strip.text.y = element_blank())

fig("tmp", w=3, h=3)

# %% --------

total_likelihoods |> 
    group_by(param_id) |> 
    left_join(generating, by="param_id", suffix=c("_fit", "_true")) |>
    filter(base_precision_true == 0.04) |> 
    group_by(param_id, cost_fit, attention_factor_fit) |> 
    slice_max(logp) |> 
    group_by(param_id) |> 
    mutate(logp = logp - max(logp)) |> 
    ggplot(aes(attention_factor_fit, cost_fit, fill=logp)) +
    geom_tile() +
    geom_point(aes(x=attention_factor_true, y=cost_true), 
               shape='+', size=3, color='red') +
    scale_fill_viridis_c() +
    facet_grid(fct_rev(factor(cost_true)) ~ attention_factor_true, scales="free") +
    labs(x="Fitted Attention Factor (θ)", 
         y="Fitted Sampling Cost (c)",
         fill="Log Likelihood") +
    theme(aspect.ratio = 1, 
          strip.text.x = element_blank(), 
          strip.text.y = element_blank())

fig("recovery-heatmap-cost", w=10, h=5)

# %% --------

total_likelihoods |> 
    group_by(param_id, attention_factor) |> 
    slice_max(logp) |> 
    left_join(generating, by=c("param_id"), suffix=c("_fit", "_true")) |> 
    ggplot(aes(attention_factor_fit, logp, color=factor(attention_factor_true))) +
    geom_point() +
    geom_abline(slope=1, intercept=0) +
    scale_color_brewer(palette="Set1") +
    facet_grid(cost_true ~ base_precision_true)
    # gam_fit(k=3, color=RED, fill=RED, alpha=0.1) +
    # gridlines +
    # theme(aspect.ratio = 1)

fig("recovery-attention",w=7)
