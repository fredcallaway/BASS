using Distributed
using ProgressMeter

@everywhere using Revise
@everywhere includet("qualitative_fitting.jl")

# %% ==================== Simulate models ====================
@everywhere version = "v3"
mkpath("tmp/qualitative/$version/")

box = Box(
    base_precision = (.0005, .01, :log),
    attention_factor = (0, 1),
    cost = (.0001, .002, :log),
    confidence_slope = (0, 0.1),
    prior_mean = (-1, 0),
)
serialize("tmp/qualitative/$version/box", box)

candidates = map(Iterators.take(SobolSeq(n_free(box)), 10000)) do x
    BDDM(;box(x)...)
end;

mkpath("tmp/qualitative/$version/sims")
@time write_base_sim()
@everywhere base_sim = deserialize("tmp/qualitative/$version/sims/base")

# %% --------

results = @showprogress pmap(enumerate(candidates)) do (i, m)
    sim_df = if isfile("tmp/qualitative/$version/sims/$i")
         load_sim(i)
    else
        x = make_frame(simulate_dataset(m, trials));
        write_sim(i, x)
        x
    end
    @. sim_df.val1 = round(sim_df.val1 * val_σ + val_μ; digits=2)
    @. sim_df.val2 = round(sim_df.val2 * val_σ + val_μ; digits=2)
    choice_fit = coeftable(fit_choice_model(sim_df))
    rt_fit = coeftable(fit_rt_model(sim_df))
    (; choice_fit, rt_fit)
end;
length(results)

# %% ==================== Compare to model fit to human ====================

human_fit_choice = fit_choice_model(human_df)
human_coef_choice = coef(human_fit_choice)
human_err_choice = stderror(human_fit_choice)

human_fit_rt = fit_rt_model(human_df)
human_coef_rt = coef(human_fit_rt)
human_err_rt = stderror(human_fit_rt)

select_choice = [
    "(Intercept)",
    "rel_value",
    "avg_value",
    "rel_conf",
    "rel_value & avg_conf",
    "avg_value & rel_conf",
    "avg_value & prop_first_presentation"
]
choice_idx = coeftable(human_fit_choice).rownms .∈ [select_choice]

select_rt = [
    "(Intercept)",
    "abs_rel_value",
    "avg_value",
    "prop_first_presentation",
    "avg_conf",
    "rel_conf",
    "rel_value & prop_first_presentation"
]
rt_idx = coeftable(human_fit_rt).rownms .∈ [select_rt]

loss = map(results) do (choice_fit, rt_fit)
    choice_loss = sum(((choice_fit.cols[1][choice_idx] .- human_coef_choice[choice_idx]) ./ human_err_choice[choice_idx]) .^ 2)
    rt_loss = sum(((rt_fit.cols[1][rt_idx] .- human_coef_rt[rt_idx]) ./ human_err_rt[rt_idx]) .^ 2)
    choice_loss + rt_loss
    #choice_loss
end
best = argmin(loss)
@show loss[best]
candidates[best]

# %% --------
rt_intercepts = map(results) do (choice_fit, rt_fit)
    rt_fit.cols[1][1]  # intercept Coef.
end
quantile(rt_intercepts)
coef(human_fit_rt)[1]

# %% --------
Table(
    predictor=coeftable(human_fit_choice).rownms,
    human=coef(human_fit_choice),
    model=results[best].choice_fit.cols[1],
    loss= choice_idx .* ((results[best].choice_fit.cols[1] .- human_coef_choice) ./ human_err_choice) .^ 2
)

Table(
    predictor=coeftable(human_fit_rt).rownms,
    human=coef(human_fit_rt),
    model=results[best].rt_fit.cols[1],
    loss= rt_idx .* ((results[best].rt_fit.cols[1] .- human_coef_rt) ./ human_err_rt) .^ 2
)

# %% --------
sim_df = make_frame(simulate_dataset(candidates[best], trials; ndt=500))

@. sim_df.val1 = round(sim_df.val1 * val_σ + val_μ; digits=2)
@. sim_df.val2 = round(sim_df.val2 * val_σ + val_μ; digits=2)
# recompute loss
@show loss[best]

# DOESN"T INCLUDE RT
#xx = coeftable(fit_choice_model(sim_df))
#new_loss = sum(((xx.cols[1][choice_idx] .- human_coef_choice[choice_idx]) ./ human_err_choice[choice_idx]) .^ 2) +
#@show new_loss

sim_df |> CSV.write("results/qualitative_sim_$version.csv")

# %% --------
function prepare_frame(df)
    Table(
        # subject = categorical(df.subject),
        choice = df.choice,
        rt = df.pt1 .+ df.pt2  .- rt_μ,
        # rt = @.((df.pt1 + df.pt2  - rt_μ) / 2rt_σ),
        rel_value = (df.val1 .- df.val2) ./ 10,
        rel_conf = df.conf1 .- df.conf2,
        avg_value = demean(df.val1 .+ df.val2) ./ 20,
        avg_conf = demean(df.conf1 .+ df.conf2),
        prop_first_presentation = (df.pt1 ./ (df.pt1 .+ df.pt2)) .- 0.5,
        conf_bias = demean(2 .* getindex.([conf_bias], df.subject))
    )
end
T = prepare_frame(sim_df)
choice_formula = @formula(choice==1 ~ rel_value)

fit(GeneralizedLinearModel, choice_formula, prepare_frame(sim_df), Bernoulli())
fit(GeneralizedLinearModel, choice_formula, prepare_frame(human_df), Bernoulli())

# %% --------

sim_df = simulate_dataset(candidates[best], trials)
@time sim_df |> CSV.write("tmp/test.csv")
@time serialize("tmp/test", sim_df)

# R = Table(results)
serialize("tmp/qualitative_apr9", (;box, results))
# %% --------

using GaussianProcesses
box, R = deserialize("tmp/qualitative_feb7")
X = reduce(hcat, Iterators.take(SobolSeq(n_free(box)), 40000))
y = R.rt_abserr

idx = 1:5000
X = X[:, idx]
y = y[idx]

# %% --------
train_idx = 1:1000
kernel = SEArd(ones(size(X, 1)), -3.)
gp = GP(X[:, train_idx], y[train_idx], MeanConst(mean(y)), kernel)
optimize!(gp)

# gp = GP(X, y, gp0.mean, gp0.kernel)
# yhat, yvar = predict_f(gp, X)
# %% --------
# test_idx = 2000:3000
test_idx = 1000:5000

yhat, yvar = predict_f(gp, X[:, test_idx])
mean(@. abs(yhat - y[test_idx]))
mean(y[test_idx])

yhat, yvar = predict_f(gp, X[:, train_idx])
mean(abs.(yhat .- y[train_idx]))

# %% --------

function find_minimum(gp; restarts=20)
    best_val = Inf; best_x = nothing
    for i in 1:restarts
        res = optimize(rand(4)) do x
            predict_f(gp, reshape(x, n_free(box), 1))[1][1]
        end
        if res.minimum < best_val
            best_x = res.minimizer
            best_val = res.minimum
        end
    end
    best_x, best_val
end

best_x, best_y = find_minimum(gp)

# %% --------
rng = 0:0.05:1
marginals = map(1:n_free(box)) do i
    map(rng) do x_target
        # LBFGS(), autodiff=:forward
        init = copy(best_x)
        deleteat!(init, i)
        res = optimize(init) do x_other
            x = [x_other[1:i-1]; x_target; x_other[i:end]]
            predict_f(gp, reshape(x, n_free(box), 1))[1][1]
        end
        res.minimum
    end
end

# %% --------
figure() do
    ps = map(collect(pairs(box.dims)), marginals) do (name, d), x
        # name, d = first(collect(pairs(box.dims)))
        maybelog = :log in d ? (:log,) : ()
        plot(rescale(d, rng), x, xaxis=(string(name), maybelog...))
    end
    plot(ps..., size=(600, 600))
end

# %% --------

map(candidates[partialsortperm(loss, 1:100)]) do m
    (;m.base_precision,
     m.attention_factor,
     m.cost,
     m.confidence_slope,
     m.prior_mean)
end |> Table


