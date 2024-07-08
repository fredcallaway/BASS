include("base.jl")
include("regressions.jl")

# %% --------

R"""
source("base.r")
FIGS_PATH = "figs/sensitivity/"
# human = read_csv("data/Study1Coefficients.csv") %>%
#     rename(term=`...1`, estimate=Estimate, std_error=`Std. Error`)
"""

function load_frame(model, analysis=:choice)
    results = deserialize("results/sensitivity/$model")

    df = flatmap(results) do x
        map(getfield(x, analysis)) do reg
            (;x.prm..., x.accuracy, reg...)
        end
    end |> DataFrame
    df.model .= model[3:end]
    df
end

# %% ==================== study 1 ====================


model1 = mapreduce(load_frame, vcat, ["1-main", "1-biased_mean", "1-zero_mean", "1-flat_prior"])

data1 = load_human_data(1)
human1 = DataFrame(fit_regressions(DataFrame(make_frame(data1)); study=1).choice)

@rput model1 human1

R"""
model_pal = scale_colour_manual(values=c(
    main = BLUE,
    flat_prior = GRAY,
    zero_mean = RED,
    biased_mean = PURPLE,
    nometa = YELLOW
), aesthetics=c("fill", "colour"), name="")

bullseye = function(data) list(
    geom_point(data=data, size=3, color="black"),
    geom_point(data=data, size=1.7, color="white"),
    geom_point(data=data, size=.7, color="black")
)

human_wide1 = human1 %>%
    select(term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)

model1 %>%
    filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, base_precision, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    select(model, `spdfirst:savV`, spdfirst) %>%
    ggplot(aes(spdfirst, `spdfirst:savV`)) +
    geom_point(mapping=aes(color=model), size=.5) +
    bullseye(human_choice_wide) +
    labs(x="Relative Presentation Duration", y="Relative Presentation\nby Overall Value") +
    theme_classic() +
    model_pal + no_legend

fig("duration_interaction", w=4, pdf=T)
"""

# %% ==================== study 2 ====================

data2 = load_human_data(2)
reg2 = fit_regressions(DataFrame(make_frame(data2)); study=2)
human_choice = DataFrame(reg2.choice)
human_rt = DataFrame(reg2.rt)

model_choice, model_rt = map([:choice, :rt]) do analysis
    mapreduce(vcat, ["2-main", "2-biased_mean", "2-zero_mean", "2-nometa"]) do name
        df = load_frame(name, analysis)
        if "subjective_slope" ∉ names(df)
            df.subjective_slope .= NaN
            df.subjective_offset .= NaN
        end
        df
    end
end

@rput model_choice model_rt human_choice human_rt

human_accuracy = reg2.accuracy
@rput human_accuracy
# %% --------

R"""
human_choice_wide = human_choice %>%
    select(term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)

model_choice %>%
    # filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, base_precision, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    select(model, `ConfDif:savV`, ConfDif) %>%
    ggplot(aes(ConfDif, `ConfDif:savV`)) +
    geom_point(mapping=aes(color=model), size=.5) +
    bullseye(human_choice_wide) +
    labs(x="Relative Confidence", y="Relative Confidence\nby Overall Value") +
    theme_classic() +
    model_pal
    # +
    # scale_colour_manual(values=c(
    #     main=BLUE,
    #     nometa=ORANGE
    # ), aesthetics=c("fill", "colour"), name="")

    # facet_wrap(~model)

fig("confidence_interaction", w=4, pdf=T)
"""

µ, σ = empirical_prior(data2)

R"""
D = model_choice %>%
    # filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, base_precision, prior_mean, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    mutate(prior_mean = prior_mean / $µ)


ggplot(filter(D, model != "nometa"), aes(ConfDif, `ConfDif:savV`)) +
    geom_point(mapping=aes(color=prior_mean), size=.5) +
    geom_point(data=filter(D, model == "nometa"), size=.5, color=YELLOW) +
    bullseye(human_choice_wide) +
    theme_classic() +
    scale_color_gradient(
      low = RED,
      high = BLUE,
    ) + labs(color="Prior Mean")


fig("confidence_interaction_altr", w=4, pdf=T)
"""

R"""

confidence_consistency = XXX
model_choice %>%
    filter(model %in% c("main", "nometa")) %>%
    filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, base_precision, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    ggplot(aes(fstosnd, `fstosnd:totalConfidence`)) +
    geom_point(mapping=aes(color=model), size=.3, alpha=1) +
    bullseye(human_choice_wide) +
    labs(x="Relative Value", y="Relative Value\nby Overall Confidence") +
    theme_classic() +
    model_pal

fig("confidence_consistency", w=4,)
"""


R"""

confidence_rt = XXX
human_wide = human_rt %>%
    select(term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)

model_rt %>%
    filter(model %in% c("main", "nometa")) %>%
    # filter(abs(accuracy - human_accuracy) < .05) %>%
    group_by(model) %>%
    # filter((model == "nometa") | (base_precision + 5 * confidence_slope > .01)) %>%
    select(model, base_precision, term, estimate, std_error, accuracy) %>%
    filter(term == "totalConfidence") %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    ggplot(aes(accuracy, totalConfidence)) +
    geom_point(mapping=aes(color=model), size=.5) +
    # geom_point(data=human_wide, size=4, shape="+", color="green") +
    geom_point(data=human_wide, size=2, color="black") +
    geom_point(data=human_wide, size=1, color="white") +
    geom_point(data=human_wide, size=.3, color="black") +
    geom_hline(yintercept=0) +
    labs(y="overall confidence → RT", x="choice consistency") +
    theme_classic() +
    model_pal



fig("confidence_rt", w=4, h=3, pdf=T)
"""

# %% --------

R"""
model_rt %>%
    filter(model %in% c("nometa")) %>%
    filter(term == "totalConfidence") %>%
    arrange(estimate) %>%
    tibble %>%
    select(estimate, std_error)
"""



# %% --------

d2 = DataFrame(make_frame(data2))
@rput d2
# %% --------

model = @chain model_rt begin
    @rsubset :model == "main"
    @rsubset :term == "totalConfidence"
    @orderby -:estimate
    @select Between(1, 6)
    BDDM(;first(eachrow(_))...)
end




m2 = DataFrame(make_sim(model, data2; repeats=10))
@rput m2
mfit = fit_regressions(m2; study=2)
mfit.rt[5]

# %% --------

R"""
accuracy = d2 %>%
    filter(val1 != val2) %>%
    with(mean((val1 > val2) == (choice == 1)))
"""

@rget accuracy

accuracy



R"""
m2 %>%
    filter(abs(val1 - val2) > 1) %>%
    mutate(correct = (val1 > val2) == (choice == 1), sumConf = conf1 + conf2) %>%
    ggplot(aes(sumConf, 1*correct)) +
    linear_fit(formula = y ~ x + I(x^2)) +
    points()

fig()
"""

# %% --------

R"""
model_rt %>%
    filter(model == "main") %>%
    filter(term == "totalConfidence") %>%
    # filter(estimate > 0) %>%
    select(estimate, std_error, 1:4) %>%
    arrange(-estimate)


"""


R"""

    filter(confidence_slope < .02, base_precision < .02) %>%
    ggplot(aes(base_precision, confidence_slope, z=estimate)) +
    stat_summary_2d(bins=10) +
    scale_fill_continuous_diverging(name="effect") +
    # geom_point(mapping=aes(color=estimate), size=.5) +
    facet_wrap(~model)

fig(w=4)
"""

R"""
model_rt %>%
    filter(term == "totalConfidence") %>%
    select(-c(prior_mean, prior_precision)) %>%
    pivot_longer(c(base_precision, confidence_slope, attention_factor, cost), names_to="parameter", values_to="param") %>%
    ggplot(aes(param, estimate, color=model)) +
    geom_hline(mapping=aes(yintercept=estimate),
        data=filter(human_rt, term == "totalConfidence")
    ) +
    facet_grid(term~parameter, scales="free") +
    geom_point(size=.1, alpha=.1) +
    gam_fit() +
    scale_colour_manual(values=c(
        main=BLUE,
        nometa=ORANGE
    ), aesthetics=c("fill", "colour"), name="")

fig(w=8)
"""

# %% ==================== other stuff ====================


R"""
df %>%
    fctrize(model, levels=c("flat_prior", "main", "biased_mean", "zero_mean")) %>%
    ggplot(aes(model, estimate, color=model)) +
    stat_mean_and_quantiles(rng=1) +
    # geom_hline(yintercept=0) +
    no_legend +
    coord_flip() +
    geom_hline(mapping=aes(yintercept=estimate), data=human) +
    facet_wrap(~term, scales="free_x") +
    gridlines + xlab("")

fig(w=5, h=4)
"""


R"""

model_choice %>%
    # filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, base_precision, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    select(model, `ConfDif:savV`, ConfDif)

"""
R"""

model_choice %>%
    filter(model == "main") %>%
    pivot_longer(c(base_precision, confidence_slope, attention_factor, cost), names_to="parameter", values_to="param") %>%
    filter(term == "ConfDif:savV") %>%
    ggplot(aes(param, estimate, color=model)) +
    geom_hline(mapping=aes(yintercept=estimate), data=filter(human_choice, term == "ConfDif:savV")) +
    facet_wrap(~parameter, scales="free", nrow=1) +
    geom_point(size=.1, alpha=.1)
    # gam_fit()

fig(w=7)
"""