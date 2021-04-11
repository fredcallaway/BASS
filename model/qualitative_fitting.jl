include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("box.jl")
# include("binning.jl")
using GLM
using ProgressMeter
using Sobol
using Serialization
using CSV

function make_frame(data)                                                                                                   
   map(data) do d                                                                                                          
       val1, val2 = d.value                                                                                                
       conf1, conf2 = d.confidence                                                                                         
       pt1 = round(sum(d.presentation_duration[1:2:end]); digits=3)                                                        
       pt2 = round(sum(d.presentation_duration[2:2:end]); digits=3)                                                        
       (;d.subject, val1, val2, conf1, conf2, pt1, pt2, d.choice)                                                                     
   end |> Table                                                                                                            
end    

data = load_human_data()
human_df = make_frame(data)
trials = repeat(prepare_trials(Table(data); dt=.025), 100);

rt_μ, rt_σ = juxt(mean, std)(human_df.pt1 + human_df.pt2)
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

function fit_choice_model(df)
    T = Table(
        # subject = categorical(df.subject),
        choice = df.choice,
        rt = @.((df.pt1 + df.pt2  - rt_μ) / 2rt_σ),
        # rt = zscore(df.pt1 .+ df.pt2),
        rel_value = normalize(df.val1 .- df.val2),
        rel_conf = normalize(df.conf1 .- df.conf2),
        avg_value = zscore(df.val1 .+ df.val2),
        avg_conf = zscore(df.conf1 .+ df.conf2),
        prop_first_presentation = zscore(df.pt1 ./ (df.pt1 .+ df.pt2)),
        conf_bias = getindex.([conf_bias], df.subject),
    )
    choice_formula = @formula(choice==1 ~ (rel_value + avg_value) *
        (avg_conf + conf_bias + rel_conf + prop_first_presentation));
    fit(GeneralizedLinearModel, choice_formula, T, Bernoulli())
end

function simulate_dataset(m, trials)
    map(trials) do t
        sim = simulate(m, t; save_presentation=true)
        presentation_duration = t.dt .* sim.presentation_durations
        m1, m2 = mean.(t.presentation_distributions)
        order = m1 > m2 ? :longfirst : :shortfirst
        rt = sim.time_step .* t.dt
        (;t.subject, t.value, t.confidence, presentation_duration, order, sim.choice, rt)
    end
end

function foo()
    println(4)
end

# function create_loss_function()
#     human_fit = fit_choice_model();
#     tbl = coeftable(human_fit)
#     coefs, errs, z, pz = tbl.cols[1:4]
#     selected = pz .<= .05
#     # println(tbl.rownames[selected])
#     coefs[selected], errs[selected]
# end