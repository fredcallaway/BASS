using Distributed
# nprocs() == 1 && addprocs()
@everywhere begin
    include("base.jl")
    using RCall
    using DataFrames
end
@everywhere R"library(broom)"
mkpath("results/sensitivity")
mkpath("results/sensitivity-json")

using ProgressMeter
using Sobol
function sobol(n::Int, box::Box)
    seq = SobolSeq(length(box))
    skip(seq, n)
    [box(Sobol.next!(seq)) for i in 1:n]
end

@everywhere df2tuples(df) = map(NamedTuple, eachrow(df))

@everywhere function fit_regressions(df; study)
    @rput df
    R"""
    df$SubNum <- factor(df$subject)
    df$isfirstIchosen <- as.numeric(df$choice ==1)
    df$sVD <- scale(abs(df$val1 -df$val2)/10, scale=FALSE, center=TRUE)
    df$savV <- scale((df$val1+df$val2)/20, scale=FALSE, center=TRUE)
    df$RT <- (df$pt1+df$pt2)
    df$cRT <- scale(df$RT, scale=FALSE, center=TRUE)
    df$sfstItemV <- scale(df$val1, scale=FALSE, center=TRUE)
    df$ssndItemVal <- scale(df$val2, scale=FALSE, center=TRUE)
    df$fstosnd <- (df$val1 -df$val2)/10
    df$spdfirst<- (df$pt1/(df$pt1+df$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)
    df$cinitpres1 <- scale(df$initpresdur1, scale=FALSE, center = TRUE)
    """
    if study == 1
        R"""
        choice_formula = isfirstIchosen ~ fstosnd +  spdfirst + cRT + savV + spdfirst:savV
        rt_formula = log(RT*1000)~ sVD + savV
        """
    elseif study == 2
        R"""
        for (i in levels(df$SubNum) ) {
          df$ConfBias[df$SubNum==i]  <-  mean(c(df$conf1[df$SubNum==i], df$conf2[df$SubNum==i]))
        }

        df$cConfBias <- scale(df$ConfBias, scale=FALSE, center= TRUE)
        df$sumConfidence <- df$conf1+df$conf2
        df$totalConfidence <- (df$sumConfidence)/2 - df$ConfBias
        df$cfstConfidence <- scale(df$conf1, scale=FALSE, center = TRUE)
        df$csndConfidence <- scale(df$conf2, scale=FALSE, center = TRUE)
        df$ConfDif <- df$conf1 - df$conf2

        choice_formula = isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence + fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV
        rt_formula = log(RT*1000) ~ sVD + savV + ConfDif + totalConfidence
        """
    end
    R"""
    choice_fit = tidy(glm(choice_formula, data = df, family=binomial(link='logit')))
    rt_fit = tidy(lm(rt_formula, data = df))
    accuracy = d2 %>%
        filter(val1 != val2) %>%
        with(mean((val1 > val2) == (choice == 1)))
    """
    (;choice = df2tuples(rcopy(R"choice_fit")),
      rt = df2tuples(rcopy(R"rt_fit")),
      accuracy = rcopy(R"accuracy")
      )
end

function run_sensitivity(name, data, box; N=1000)
    study = parse(Int, name[1])
    results = @showprogress name pmap(sobol(N, box)) do prm
        if ismissing(get(prm, :subjective_offset, nothing))
            subjective_offset = prm.confidence_slope * mean(flatten(data.confidence))
            prm = (;prm..., subjective_offset)
        end
        model = BDDM(;prm...)
        df = DataFrame(make_sim(model, data; repeats=30))



        (;prm, fit_regressions(df; study)...)
    end
    serialize("results/sensitivity/$name", results)
    write("results/sensitivity-json/$name.json", json(results))
end

# %% ==================== study 1 ====================


data1 = load_human_data(1)
µ, σ = empirical_prior(data1)
box1 = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)

run_sensitivity("1-main", data1, box1)
run_sensitivity("1-biased_mean", data1, update(box1, prior_mean=(µ/2, µ)))
run_sensitivity("1-zero_mean", data1, update(box1, prior_mean = 0.))
run_sensitivity("1-flat_prior", data1, update(box1, prior_precision = 1e-8))

# %% ==================== study 2 ====================

data2 = load_human_data(2)
µ, σ = empirical_prior(data2)

box2 = Box(
    base_precision = (.0001, .1, :log),
    confidence_slope = (.001, .1, :log),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1 / σ^2,
    subjective_offset = 0.,
    subjective_slope = 1.,
)

run_sensitivity("2-main", data2, box2)
run_sensitivity("2-biased_mean", data2, update(box2, prior_mean = (µ/2, µ)))
run_sensitivity("2-zero_mean", data2, update(box2, prior_mean = 0.))
run_sensitivity("2-nometa", data2, update(box2, subjective_slope = 0, subjective_offset = missing))
