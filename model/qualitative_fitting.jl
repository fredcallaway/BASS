include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("box.jl")
# include("binning.jl")
using GLM
using Sobol
using Serialization
using CSV

function make_frame(data)
   map(data) do d
       val1, val2 = d.value
       conf1, conf2 = d.confidence
       pt1 = round(sum(d.presentation_duration[1:2:end]); digits=3)
       pt2 = round(sum(d.presentation_duration[2:2:end]); digits=3)
       (;d.subject, val1, val2, conf1, conf2, pt1, pt2, d.choice, d.rt)
   end |> Table
end

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

rt_μ, rt_σ = juxt(mean, std)(human_df.pt1 + human_df.pt2)
val_μ, val_σ = juxt(mean, std)(flatten(data.value))
conf_bias = map(group(x->x.subject, human_df)) do x
    mean([x.conf1; x.conf2])
end

demean(x) = x .- mean(x)
normalize(x) = x ./ std(x)
zscore(x) = x .- mean(x) ./ std(x)

# function fit_choice_model(df)
#     T = Table(
#         # subject = categorical(df.subject),
#         choice = df.choice,
#         rel_value = normalize(df.val1 .- df.val2),
#         rel_conf = normalize(df.conf1 .- df.conf2),
#         avg_value = zscore(df.val1 .+ df.val2),
#         avg_conf = zscore(df.conf1 .+ df.conf2),
#         prop_first_presentation = df.pt1 ./ (df.pt1 .+ df.pt2) .- 0.5,
#         conf_bias = getindex.([conf_bias], df.subject),
#     )
#     choice_formula = @formula(choice==1 ~ 
#         rel_value * avg_conf +
#         avg_value * rel_conf +
#         avg_value * prop_first_presentation
#     )
#     fit(GeneralizedLinearModel, choice_formula, T, Bernoulli())
# end

function prepare_frame(df)
    Table(
        # subject = categorical(df.subject),
        choice = df.choice,
        rt = df.pt1 .+ df.pt2  .- rt_μ,
        log1000rt = log.(1000 .* (df.pt1 .+ df.pt2)),
        # rt = @.((df.pt1 + df.pt2  - rt_μ) / 2rt_σ),
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
    if STUDY == 3       
        formula = @formula(choice==1 ~ rel_value + avg_value + rel_conf + avg_conf + prop_first_presentation  + rt +
            rel_value & avg_conf + rel_conf & avg_value + prop_first_presentation & avg_value);
    elseif STUDY == 2
        # TODO: add this @formulat(choice==1 ~ rel_value + prop_first_presentation * (avg_value + rel_value) + rt)
        formula = @formula(choice==1 ~ (rel_value + avg_value) * prop_first_presentation + rt);
    else
        error("Bad STUDY")
    end 
    fit(GeneralizedLinearModel, formula, prepare_frame(df), Bernoulli())
end

function fit_rt_model(df)
    if STUDY == 3
        #formula = @formula(log1000rt ~ abs_rel_value + avg_value + rel_conf + avg_conf + rel_value)
        formula = @formula(log1000rt ~ abs_rel_value + avg_value + rel_conf + avg_conf + rel_value + prop_first_presentation + 
            rel_value & prop_first_presentation)
    elseif STUDY == 2
        # TODO
        formula = @formula(log1000rt ~ (abs_rel_value + rel_value + avg_value) * prop_first_presentation)
    else
        error("Bad STUDY")
    end 
    fit(LinearModel, formula, prepare_frame(df))
end

function simulate_dataset(m, trials; ndt=0)
    map(trials) do t
        sim = simulate(m, SimTrial(t); save_presentation=true)
        presentation_duration = t.dt .* sim.presentation_durations
        m1, m2 = mean.(t.presentation_distributions)
        order = m1 > m2 ? :longfirst : :shortfirst
        rt = sim.time_step .* t.dt + ndt
        (;t.subject, t.value, t.confidence, presentation_duration, order, sim.choice, rt)
    end
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

# function create_loss_function()
#     human_fit = fit_choice_model();
#     tbl = coeftable(human_fit)
#     coefs, errs, z, pz = tbl.cols[1:4]
#     selected = pz .<= .05
#     # println(tbl.rownames[selected])
#     coefs[selected], errs[selected]
# end