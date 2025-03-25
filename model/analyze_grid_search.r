# %% --------

source("base.r")
library(tidyverse)

# %% --------

full_results <- read_csv("results/grid/2025-03-20.csv")

# %% --------


full_results %>%
    group_by(subject) %>%
    slice_max(logp) %>% 
    ggplot(aes(base_precision, cost)) +
    geom_jitter(alpha=0.4, width=0.0005, height=0.0005)

fig()

# %% --------

df <- full_results |>
    group_by(cost, base_precision) |>
    summarize(
        logp = sum(logp),
        sd = sqrt(sum(std ^ 2))
    ) |> 
    drop_na(sd)

df %>% ungroup() %>% slice_max(logp)
# %% --------

df |> 
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
