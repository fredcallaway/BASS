source("~/lib/base.r")
df = read_csv("results/v7-1064-unyoked.csv")


bin = function(x, breaks=4) {
    cut(x, breaks, ordered_result = TRUE)
}


df = df %>% mutate(
    rt = pt1 + pt2,
    val_diff = val1 - val2,
    conf_diff = conf1 - conf2,
    choose_first = as.integer(choice == 1),
    order = as.factor(order),
    subject = as.factor(subject),
    val_sum = val1 + val2,
    conf_sum = conf1 + conf2
)

df = df %>% mutate(across(ends_with("diff") | ends_with("sum"), bin, .names="{col}_bin"))

geom_logistic = geom_smooth(method = "glm", se = F, method.args = list(family = "binomial"))
# %% --------
ggplot(data=df, mapping=aes(x=pt1 - pt2, y=choose_first)) +
    geom_logistic +
    stat_summary_bin() +
    ylim(0,1)
fig("choice ~ conf")

# %% --------
ggplot(data=df, mapping=aes(x=pt1 - pt2, y=choose_first, color=val_sum_bin)) +
    geom_logistic +
    ylim(0,1)
fig("choice ~ conf")

# %% --------
ggplot(data=df, mapping=aes(x=conf_diff, y=choose_first)) +
    geom_logistic +
    stat_summary() +
    ylim(0,1)
fig("choice ~ conf")

# %% --------

ggplot(data=df, mapping=aes(x=conf_diff, y=choose_first, color=val_sum_bin)) +
    geom_logistic +
    ylim(0,1)
fig("choice ~ conf * val_sum")

# %% --------
ggplot(df, aes(x=conf_diff, y=choose_first)) + 
    stat_summary()
    
fig()p
# %% --------
ggplot(df, aes(x=pt1 - pt2, y=choose_first, color=val_diff, group=cut(val_diff)) +
    stat_summary_bin()
    
fig()

# %% --------
