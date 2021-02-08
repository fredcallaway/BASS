function fit_gp(xs, nll)
    X = reduce(hcat, xs)
    kernel = SEArd(ones(size(X, 1)), -3.)

    # first fit on half the data to estimate out of sample prediction accuracy
    gp = GP(X[:, 1:2:end], nll[1:2:end], MeanConst(mean(nll)), kernel)
    optimize!(gp)
    yhat, yvar = predict_f(gp, X[:, 2:2:end])
    mae = mean(abs.(yhat .- nll[2:2:end]))
    println("MAE on held out data: ", round(mae; digits=2))
    println("MAE of mean prediction: ", round(mean(abs.(mean(nll) .- nll)); digits=2))

    # use all the data for further analysis
    gp = GP(X, nll, MeanConst(mean(nll)), kernel)
    optimize!(gp)
    gp
end

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
# %% --------

function fillmissing2(target, repls)
    @assert sum(ismissing.(target)) == length(repls)
    repls = Iterators.Stateful(repls)
    x = Array{Float64}(undef, length(target))
    for i in eachindex(target)
        if ismissing(target[i])
            x[i] = first(repls)
        else
            x[i] = target[i]
        end
    end
    x
end

function find_minimum_conditional(gp, condition; restarts=20)
    best_val = Inf; best_x = nothing
    for i in 1:restarts
        res = optimize(rand(sum(ismissing.(condition)))) do x_opt
            x = fillmissing2(condition, x_opt)
            predict_f(gp, reshape(x, n_free(box), 1))[1][1]
        end
        if res.minimum < best_val
            best_x = res.minimizer
            best_val = res.minimum
        end
    end
    best_x, best_val
end


# %% ==================== Fit GP and find best fitting point ====================

@unpack box, xs, results, chance = deserialize("tmp/sobol_1")
nll = -invert(results).logp
gp = fit_gp(xs, nll)
best_x, best_y = find_minimum(gp)

# %% --------
data = filter(d->d.subject == all_data.subject[1], all_data)
trials = prepare_trials(data; dt=.025)
X = reduce(hcat, xs)
predict_f(gp, X)

function get_nll(x)
    m = BDDM(;box(x)...)
    ibs_loglike(m, trials[1:2:end]; ε=.5, tol=10, repeats=100, min_multiplier=2).logp
end




# %% ==================== Plot the marginals ====================

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


figure() do
    ps = map(collect(pairs(box.dims)), marginals) do (name, d), x
        # name, d = first(collect(pairs(box.dims)))
        maybelog = :log in d ? (:log,) : ()
        plot(rescale(d, rng), x, xaxis=(string(name), maybelog...))
    end
    plot(ps..., size=(600, 600))
end

