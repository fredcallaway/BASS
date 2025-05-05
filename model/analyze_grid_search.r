# %% --------

source("base.r")
library(tidyverse)

# %% ===== load results ========================================================


full_results <- bind_rows(
    read_csv("results/grid/2025-04-07-theta-1.csv"),
    read_csv("results/grid/2025-04-07-thetalarge-1.csv")
)

# %% --------

chance <- full_results %>% 
    filter(cost == 0) %>% 
    distinct(subject, logp) %>% 
    rename(baseline = logp)

df <- full_results %>%
    left_join(chance) %>% 
    # remove any repeated evaluations
    group_by(subject, cost, base_precision, attention_factor) %>% 
    slice_head() %>% 
    ungroup %>% 
    mutate(relative_logp = logp - baseline)

# %% --------


group <- df |>
    group_by(cost, base_precision, confidence_slope, attention_factor) |>
    summarize(
        logp = sum(logp),
        relative_logp = sum(relative_logp),
        sd = sqrt(sum(std ^ 2))
    )
    # drop_na(sd)

# %% --------

group %>% 
    filter(attention_factor > 1) %>% 
    group_by(base_precision, attention_factor) %>% 
    slice_max(relative_logp) %>% 
    ungroup() %>% 
    drop_extreme(relative_logp, q_lo=0.8, q_hi=1) %>% 
    arrange(logp)

# %% --------

group |> 
    filter(base_precision > 0) %>% 
    group_by(base_precision, attention_factor) %>% 
    slice_max(relative_logp) %>% 
    ungroup() %>% 
    drop_extreme(relative_logp, q_lo=0.8, q_hi=1) %>% 
    ggplot(aes(x=attention_factor, y=base_precision, fill=relative_logp)) +
    geom_raster() +
    coord_cartesian(xlim=c(0+.05,2-.05), ylim=c(0.01, 0.1)) +
    labs(x="Attention Factor (θ)", 
        y="Baseline Precision (τ)",
        fill="Log Likelihood\n(vs. Chance)") +
    scale_fill_viridis_c() +
    no_gridlines
    
    # scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0)

fig("grid-theta-tau",w=6)

# %% --------

group |> 
    filter(base_precision > 0) %>% 
    group_by(base_precision, attention_factor) %>% 
    slice_max(relative_logp) %>% 
    ungroup() %>% 
    drop_extreme(relative_logp, q_lo=0.9, q_hi=1) %>% 
    ggplot(aes(x=attention_factor, y=base_precision, fill=relative_logp)) +
    geom_raster() +
    expand_limits(x=0) +
    coord_cartesian(expand=F) +
    labs(x="Fitted Attention Factor (θ)", 
        y="Fitted Baseline Precision (τ)",
        fill="Mean LL\n(vs. chance)") +
    scale_fill_viridis_c() +
    no_gridlines
    # scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0)

fig("grid-theta-tau-clipped",w=6)


# %% --------

group %>% ungroup() %>% distinct(attention_factor)

group %>% 
    ggplot(aes(x=base_precision, y=attention_factor, fill=logp)) +
    geom_raster() +
    expand_limits(x=c(0, 0.1), y=c(0, 0.1)) +
    facet_wrap(~cost)
    
fig(w=7, h=7)
    

# %% --------

group |> 
    filter(base_precision > 0) %>% 
    filter(attention_factor == 1) %>% 
    filter(cost != 0) %>% 
    group_by(base_precision, cost) %>% 
    slice_max(relative_logp) %>% 
    ungroup() %>% 
    # drop_extreme(relative_logp, q_lo=0.9, q_hi=1) %>% 
    ggplot(aes(x=cost, y=base_precision, fill=relative_logp)) +
    geom_raster() +
    expand_limits(x=0) +
    coord_cartesian(expand=F) +
    labs(x="Fitted Cost (c)", 
        y="Fitted Baseline Precision (τ)",
        fill="Mean LL\n(vs. chance)") +
    scale_fill_viridis_c() +
    no_gridlines
    # scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0)

fig("grid-cost-tau",w=6)


# %% --------

df %>%
    group_by(subject) %>%
    slice_max(logp) %>% 
    count(base_precision, cost) %>% 
    ggplot(aes(base_precision, cost, fill=n)) +
    geom_tile()
    # geom_jitter(alpha=0.4, width=0.0001, height=0.0001)

fig()

# %% --------

df %>%
    ggplot(aes(x=base_precision, y=cost, fill=relative_logp)) +
    geom_raster() + 
    facet_wrap(~subject) +
    scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0)
fig(w=10, h=10)

# %% --------

group <- df |>
    group_by(cost, base_precision, confidence_slope) |>
    summarize(
        logp = sum(logp),
        relative_logp = sum(relative_logp),
        sd = sqrt(sum(std ^ 2))
    ) |> 
    drop_na(sd)

group %>% ungroup() %>% slice_max(logp)
# %% --------
group |> 
    group_by(base_precision, atten) |>
    slice_max(logp) %>% 
    ggplot(aes(x=base_precision, y=cost, fill=logp)) +
    geom_raster()

fig()
# %% --------

df |> 
    ggplot(aes(x=base_precision, y=attention_factor, fill=logp)) +
    facet_grid(version~cost, labeller="label_both") +
    geom_raster()

fig(w=10, h=3)

# %% --------

best_cost <- df |>
    group_by(version) |>
    slice_max(logp) |>
    with(unique(cost))

df |> 
    filter(cost == best_cost) |>
    ggplot(aes(x=base_precision, y=attention_factor, fill=logp)) +
    facet_wrap(~version, labeller="label_both") +
    labs(x = "τ (precision)", y = "θ (attention weight)") +
    geom_raster()
fig(w=6)
# %% --------

df |> 
    filter(cost == best_cost) |> 
    ggplot(aes(x=attention_factor, y=logp, color=factor(base_precision))) +
    geom_line() +
    geom_ribbon(aes(ymin=logp-sd, ymax=logp+sd, color=NA, fill=factor(base_precision)), alpha=0.2,) +
    scale_color_viridis_d(name="τ (precision)", aesthetics=c("color", "fill")) +
    theme(
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank()
    ) +
    labs(x = "θ (attention weight)") +
    facet_wrap(~version)

fig("grid-theta-tau", w=6.5, h=2.5)
