include("base.jl")
include("regressions.jl")


# %% --------

R"""
source("base.r")
FIGS_PATH = "figs/sensitivity/"
# human = read_csv("data/Study1Coefficients.csv") %>%
#     rename(term=`...1`, estimate=Estimate, std_error=`Std. Error`)
"""

# %% --------

version = "2025-04-11"
outdir = "results/sensitivity/processed/$version"
indir = "tmp/sensitivity/$version"
mkpath(outdir)

function load_frame(model, analysis=:choice)
    results = deserialize("$indir/$model")

    df = flatmap(results) do x
        map(getfield(x, analysis)) do reg
            rt = NamedTuple{(:rt0, :rt25, :rt50, :rt75, :rt100)}(x.rt_quantiles)
            (;x.prm..., x.accuracy, reg..., rt...)
        end
    end |> DataFrame
    df.model .= model[3:end]
    df
end

# %% ===== load data ==========================================================


s1_model_choice = mapreduce(load_frame, vcat, ["1-main", "1-zero_mean", "1-flat_prior"])
reg1 = fit_regressions(DataFrame(make_frame(load_human_data(1))); study=1)
s1_human_choice = DataFrame(reg1.choice)

# %% --------

reg2 = fit_regressions(DataFrame(make_frame(load_human_data(2))); study=2)
s2_human_choice = DataFrame(reg2.choice)
s2_human_rt = DataFrame(reg2.rt)

model_choice, model_rt = map([:choice, :rt]) do analysis
    println(analysis)
    mapreduce(vcat, ["2-main", "2-zero_mean", "2-nometa"]) do name
        df = load_frame(name, analysis)
        if "subjective_offset" ∉ names(df)
            df.subjective_slope .= NaN
            df.subjective_offset .= NaN
        end
        df
    end
end

# %% ===== fitting ============================================================

m1 = DataFrames.select(s1_model_choice, :model, 1:6, :accuracy, :rt50)
m1.study .= 1

m2 = DataFrames.select(model_choice, :model, 1:6, :accuracy, :rt50)
m2.study .= 2

model = vcat(m1, m2)

human = mapreduce(vcat, enumerate((reg1, reg2))) do (i, reg)
    rt50 = reg.rt_quantiles[3]
    accuracy = reg.accuracy
    DataFrame(;study=i, h_rt50=rt50, h_accuracy=accuracy)
end

R"""
$model |> 
    left_join($human) |> 
    mutate( rt_err = (rt50 - h_rt50)/5, accuracy_err = accuracy - h_accuracy ) |> 
    mutate(loss = rt_err^2 + accuracy_err^2) |> 
    group_by(model, study, across(2:7)) |> 
    summarise(loss = mean(loss), rt_loss = mean(rt_err^2), accuracy_loss = mean(accuracy_err^2)) |> 
    group_by(model, study) |> 
    slice_min(loss, n=1) |> 
    write_csv("results/summary_fits.csv")
"""

# %% ==================== study 1 ====================

@rput s1_model_choice s1_human_choice

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

human_wide1 = s1_human_choice %>%
    select(term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)

s1_model_choice %>%
    filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, confidence_slope, cost, accuracy, term, estimate) %>%
    filter(accuracy > .55) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    select(model, `spdfirst:savV`, spdfirst) %>%
    ggplot(aes(spdfirst, `spdfirst:savV`)) +
    geom_point(mapping=aes(color=model), size=.3) +
    bullseye(human_wide1) +
    bullseye(human2) +
    # labs(x="rpd > choice", y="rpd x ov > choice") +
    theme_classic() +
    model_pal + no_legend

fig("duration_interaction", w=2.5, pdf=T)
"""

# %% ==================== study 2 ====================

data2 = load_human_data(2)
reg2 = fit_regressions(DataFrame(make_frame(data2)); study=2)
s2_human_choice = DataFrame(reg2.choice)
s2_human_rt = DataFrame(reg2.rt)

model_choice, model_rt = map([:choice, :rt]) do analysis
    println(analysis)
    mapreduce(vcat, ["2-main", "2-zero_mean", "2-nometa"]) do name
        df = load_frame(name, analysis)
        if "subjective_offset" ∉ names(df)
            df.subjective_slope .= NaN
            df.subjective_offset .= NaN
        end
        df
    end
end

@rput model_choice model_rt s2_human_choice s2_human_rt
human_accuracy = reg2.accuracy
@rput human_accuracy human_rt50

s2_human_rt
R"""
human2 = bind_rows(
    mutate(s2_human_choice, term = glue("choice-{term}")),
    mutate(s2_human_rt, term = glue("rt-{term}")),
) %>% select(term, estimate) %>% pivot_wider(names_from=term, values_from=estimate)

"""

# %% --------

R"""
model2 = bind_rows(
    mutate(model_choice, term = glue("choice-{term}")),
    mutate(model_rt, term = glue("rt-{term}"))
) %>%
    select(model, confidence_slope, cost, accuracy, rt50, prior_mean, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)
"""

# %% --------

R"""
model2 %>%
    filter(accuracy > .55) |> 
    ggplot(aes(`choice-ConfDif`, `choice-ConfDif:savV`)) +
    geom_point(mapping=aes(color=model), size=.3) +
    bullseye(human2) +
    theme_classic() +
    expand_limits(x=0, y=0) +
    model_pal + no_legend
    # +
    # scale_colour_manual(values=c(
    #     main=BLUE,
    #     nometa=ORANGE
    # ), aesthetics=c("fill", "colour"), name="")

    # facet_wrap(~model)

fig("confidence_interaction", w=2.5, pdf=T)
"""

# %% --------


R"""
model2 %>%
    filter(model %in% c("main", "nometa")) %>%
    filter(accuracy > .55) %>%
    mutate(accurate = accuracy - human_accuracy > 0) %>%
    ggplot(aes(`rt-totalConfidence`, `choice-fstosnd:totalConfidence`)) +
    geom_point(mapping=aes(color=model), size=.3) +
    # geom_point(data=human_wide, size=4, shape="+", color="green") +
    bullseye(human2) +
    geom_vline(xintercept=0) +
    # labs(y="overall confidence → RT", x="choice consistency") +
    theme_classic() +
    model_pal + no_legend

fig("confidence_rt", w=2.5, pdf=T)
"""

# %% --------

R"""
model_rt %>%
    filter(model %in% c("nometa")) %>%
    filter(term == "totalConfidence") %>%
    arrange(estimate) %>%
    tibble %>%
    select(estimate, std_error, rt)
"""
# %% --------


R"""
model2 %>%
    filter(model %in% c("main", "nometa")) %>%
    ggplot(aes(`choice-fstosnd`, `choice-fstosnd:totalConfidence`)) +
    geom_point(mapping=aes(color=model), size=.3, alpha=1) +
    bullseye(human2) +
    # labs(x="Relative Value", y="Relative Value\nby Overall Confidence") +
    theme_classic() +
    model_pal + no_legend

fig("confidence_consistency", w=2.5, pdf=T)
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
        data=filter(s2_human_rt, term == "totalConfidence")
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

R"""
read_csv("results/dec1/1-main.csv"b) %>%
    summarise(mean(rt > 6))
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
    geom_hline(mapping=aes(yintercept=estimate), data=filter(s2_human_choice, term == "ConfDif:savV")) +
    facet_wrap(~parameter, scales="free", nrow=1) +
    geom_point(size=.1, alpha=.1)
    # gam_fit()

fig(w=7)
"""