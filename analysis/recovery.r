source("~/lib/base.r")
df = read_csv("tmp/qualitative.csv")
h = read_csv("tmp/qualitative_human.csv")


melt_dv = function(df) {
    pivot_longer(df, c(rt, choose_best, choose_first),
                 names_to = "DV", values_to="dv")
}

# %% --------

df %>% 
    pivot_longer(c(base_precision, attention_factor, cost, confidence_slope, prior_mean),
                 names_to = "IV", values_to="iv") %>% 
    melt_dv %>% 
    ggplot(aes(iv, dv)) +
        geom_smooth() +
        geom_hline(aes(yintercept=dv), color="red", data=melt_dv(h)) +
        facet_grid(DV ~ IV, scales="free") +
        theme_article()

# %% --------
ggplot(df, aes(cost, rt)) + 
    geom_smooth() + 
    geom_vline(xintercept = c(.01, .03)) +
    scale_x_continuous(trans='log10') + 
    # scale_x_continuous(trans='log10', limits=c(.01, .03)) + 
    geom_hline(aes(yintercept=rt), color="red", data=h)
