# %% --------

source("base.r")

df <- read_csv("results/sanity_check_empirical.csv")

# %% --------

df
df |> 
    mutate(
        ov = as.vector(scale(val1 + val2)),
        rpd = midbins(as.vector(scale(pt1 - pt2, center = FALSE)), round(seq(-2, 2, length.out = 6), 2)),
        choice = as.numeric(choice == 1)
    ) |> 
    drop_na() |> 
    ggplot(aes(ov, choice, color=factor(rpd))) +
    # stat_summary_bin(fun.data=mean_cl_normal, bins=5) +
    geom_smooth(method="glm", method.args=list(family=binomial), se=FALSE) +
    # scale_color_manual(values=c("#C43C20", "#A35C89", "#775C89", "#4D5D8A", "#4E7CBE")) +
    zissou_pal +
    labs(x="Overall Value", y="Choice Probability", color="Relative Presentation") +
    facet_wrap(~attention_factor)

fig(w=7, h=5)

# %% --------

df |> 
    mutate(
        ov = as.vector(scale(val1 + val2)),
        rpd = if_else(initpresdur1 > initpresdur2, "Longer First", "Shorter First"),
        choice = as.numeric(choice == 1)
    ) |> 
    drop_na() |> 
    ggplot(aes(ov, choice, color=rpd)) +
    # stat_summary_bin(fun.data=mean_cl_normal, bins=5) +
    geom_smooth(method="glm", method.args=list(family=binomial), se=FALSE) +
    # scale_color_manual(values=c("#C43C20", "#A35C89", "#775C89", "#4D5D8A", "#4E7CBE")) +
    labs(x="Overall Value", y="Choice Probability", color="Relative Presentation") +
    facet_wrap(~attention_factor)

fig(w=7, h=5)
