include("base.jl")
using GLM

if @isdefined(STUDY)
    if myid() == 1
        @info "Setting up fitting for study $STUDY"
    end
else
    error("STUDY is not set")
end

data = load_human_data(STUDY)
human_df = make_frame(data)
trials = repeat(prepare_trials(Table(data); dt=.1), 10);

val_µ, val_σ = empirical_prior(data)
rt_µ, rt_σ = juxt(mean, std)(human_df.pt1 + human_df.pt2)
conf_bias = map(group(x->x.subject, human_df)) do x
    mean([x.conf1; x.conf2])
end

demean(x) = x .- mean(x)
normalize(x) = x ./ std(x)
zscore(x) = x .- mean(x) ./ std(x)

function prepare_frame(df)
    Table(
        # subject = categorical(df.subject),
        choice = df.choice,
        rt = df.pt1 .+ df.pt2  .- rt_µ,
        log1000rt = log.(1000 .* (df.pt1 .+ df.pt2)),
        # rt = @.((df.pt1 + df.pt2  - rt_µ) / 2rt_σ),
        rel_value = (df.val1 .- df.val2) ./ 10,
        rel_conf = df.conf1 .- df.conf2,
        abs_rel_value = demean(abs.((df.val1 .- df.val2) ./ 10)),
        avg_value = demean(df.val1 .+ df.val2) ./ 20,
        avg_conf = demean(df.conf1 .+ df.conf2),
        prop_first_presentation = (df.pt1 ./ (df.pt1 .+ df.pt2)) .- 0.5,
        conf_bias = demean(2 .* getindex.([conf_bias], df.subject))
    )
end

function fit_choice_model(df)
    if STUDY == 2
        formula = @formula(choice==1 ~ rel_value + avg_value + rel_conf + avg_conf + prop_first_presentation  + rt +
            rel_value & avg_conf + rel_conf & avg_value + prop_first_presentation & avg_value);
    elseif STUDY == 1
        #isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV + spdfirst:fstosnd
        # TODO: add this @formulat(choice==1 ~ rel_value + prop_first_presentation * (avg_value + rel_value) + rt)
        formula = @formula(choice==1 ~ (rel_value + avg_value) * prop_first_presentation + rt);
    else
        error("Bad STUDY")
    end 
    fit(GeneralizedLinearModel, formula, prepare_frame(df), Bernoulli())
end

function fit_rt_model(df)
    if STUDY == 2
        #formula = @formula(log1000rt ~ abs_rel_value + avg_value + rel_conf + avg_conf + rel_value)
        formula = @formula(log1000rt ~ abs_rel_value + avg_value + rel_conf + avg_conf + rel_value + prop_first_presentation + 
            rel_value & prop_first_presentation)
    elseif STUDY == 1
        formula = @formula(log1000rt ~ abs_rel_value + avg_value)
        #formula = @formula(log1000rt ~ (abs_rel_value + rel_value + avg_value) * prop_first_presentation)
    else
        error("Bad STUDY")
    end 
    fit(LinearModel, formula, prepare_frame(df))
end


function write_sim(i, df)
    serialize("tmp/qualitative/$version/sims/$i", (;df.pt1, df.pt2, choose_second=df.choice .== 2))
end

function load_sim(i)
    x = deserialize("tmp/qualitative/$version/sims/$i");
    df = deepcopy(base_sim)
    df.choice .= 1 .+ x.choose_second
    df.pt1 .= x.pt1
    df.pt2 .= x.pt2
    df
end

function write_base_sim()
  base_sim = make_frame(simulate_dataset(candidates[1], trials));
  base_sim.choice .= 0
  base_sim.pt1 .= NaN
  base_sim.pt2 .= NaN
  serialize("tmp/qualitative/$version/sims/base", base_sim)
end

