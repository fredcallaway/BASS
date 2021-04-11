using Distributed
@everywhere using Revise
@everywhere includet("qualitative_fitting.jl")

# %% ==================== Regress on human ====================

r_coef =  [0.18520,  3.38329, -0.06293, -0.01858, -0.01758,  0.11178,  0.15935, -0.21492,  0.23312, -0.83975,  0.03660, -0.20990,  0.09570, -0.11671,  0.62663,  0.90826, ]

human_fit = fit_choice_model(human_df)
Table(
    predictor=coeftable(human_fit).rownms,
    r=r_coef,
    julia=coef(human_fit),
    # model=results[best].cols[1]
)

# %% ==================== Sobol ====================

box = Box(
    base_precision = (.01, 1, :log),
    attention_factor = (0, 2),
    cost = (.01, .1, :log),
    confidence_slope = (0, 0.1),
    prior_mean = (-1, 1),
)

candidates = map(Iterators.take(SobolSeq(n_free(box)), 10000)) do x
    BDDM(;box(x)...)
end

# m = first(candidates)
# using SplitApplyCombine: flatten

mkdir("tmp/qualitative/sims")

@time results = @showprogress pmap(enumerate(candidates)) do (i, m)
    @time sim_df = make_frame(simulate_dataset(m, trials[1:72750]));
    # serialize("tmp/qualitative/sims/$i", sim_df)
    sim_fit = fit_choice_model(sim_df)
    coeftable(sim_fit)
end

sim_df = simulate_dataset(candidates[best], trials[1:72750])
@time sim_df |> CSV.write("tmp/test.csv")
@time serialize("tmp/test", sim_df)

# R = Table(results)
serialize("tmp/qualitative_apr9", (;box, results))
# %% --------
human_fit = fit_choice_model(human_df)
human_coef = coef(human_fit)
human_err = stderror(human_fit)

loss = map(results) do r
    sum(((r.cols[1] .- human_coef) ./ human_err) .^ 2)
end

best = argmin(loss)
loss[best]
m = candidates[best]
sim_df = make_frame(simulate_dataset(m, trials[1:72750]))
sim_df |> CSV.write("results/qualitative_sim_apr10.csv")



Table(
    predictor=coeftable(human_fit).rownms,
    human=coef(human_fit),
    model=results[best].cols[1]
)

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