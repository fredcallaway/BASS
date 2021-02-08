@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("box.jl")
end

using ProgressMeter
using Sobol
using Serialization
using CSV

dt = .025
data = filter(d->d.subject == 1064, all_data)
presentation_dists = Normal(.2/dt, .05/dt), Normal(.5/dt, .1/dt)
trials = prepare_trials(data; dt)
@everywhere trials = $trials

@everywhere function evaluate(m)
    max_rt = quantile([t.rt for t in trials], .99)
    max_rt = 3 * round(Int, max_rt)
    map(repeat(trials, 100)) do t
        st = SimTrial(t.value, t.confidence, t.presentation_times)
        sim = simulate(m, DirectedCognition(m); t=st, max_rt)

        vd = t.value[1] - t.value[2]
        choose_best = 
            vd < 0 ? float(sim.choice == 2) : 
            vd > 0 ? float(sim.choice == 1) : 
            NaN
        (
            choose_best,
            choose_first = sim.choice .== 1,
            choose_human = sim.choice .== t.choice,
            sim.rt,
            rt_abserr = abs(sim.rt - t.rt),
        )
    end |> invert |> map(mean)
end


# %% ==================== Grid ====================


box = Box(
    base_precision = (10^-4, 10^-2, :log),
    attention_factor = (0, 1),
    cost = (10^-4.5, 10^-2.5, :log),
    risk_aversion = (0, .05)
)

candidates = map(grid(10, box)) do g
    BDDM(;g...)
end

results = @showprogress pmap(evaluate, candidates);
serialize("tmp/qualitative_grid_feb7", (;box, results))

# %% --------



# %% ==================== Sobol ====================

box = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1),
    cost = (1e-5, 1e-3, :log),
    risk_aversion = (0, 1e-2)
)

candidates = map(Iterators.take(SobolSeq(n_free(box)), 40000)) do x
    BDDM(;box(x)...)
end

results = @showprogress pmap(evaluate, candidates);
R = Table(results)
serialize("tmp/qualitative_feb7", (;box, R))

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
# test_idx = 1000:2000
yhat, yvar = predict_f(gp, X[:, test_idx])
mean(abs.(yhat .- y[test_idx]))

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








