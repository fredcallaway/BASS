using GaussianProcesses
using Parameters
using Serialization
using SplitApplyCombine
using Statistics
using Optim
using Sobol

include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("likelihood.jl")
include("ibs.jl")
include("box.jl")

function fit_gp(xs, nll, nll_var; gp_mean = mean(nll))
    X = reduce(hcat, xs)
    gp_mean = MeanConst(gp_mean)
    kernel = SEArd(ones(size(X, 1)), -3.)

    log_noise = @.(log(√(nll_var)))
    train = eachindex(nll)[1:2:end]
    test = eachindex(nll)[2:2:end]

    # first fit on half the data to estimate out of sample prediction accuracy
    gp = GPE(X[:, train], nll[train], gp_mean, kernel, log_noise[train])
    optimize!(gp, domean=false, noise=false)
    yhat, yvar = predict_f(gp, X[:, test])
    rmse(preds) = √ mean((preds .- nll[test]).^2)
    # mae(preds) = mean(abs.(preds .- nll[test]))
    println("RMSE on held out data: ", rmse(yhat) |> round2)
    println("RMSE of mean prediction: ", rmse(mean(nll[train])) |> round2)
    println("RMSE of perfect prediction: ", √mean(nll_var) |> round2)
    println("RMSE scaled by prediction variance: ", 
        mean(((yhat .- nll[test]) ./ .√yvar).^2) |> round2)
    
    # use all the data for further analysis
    gp = GPE(X, nll, gp_mean, kernel, log_noise)
    optimize!(gp, domean=false, noise=false)
    gp
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

function minimize(f::Function; restarts=20)
    best_val = Inf; best_x = nothing
    for x0 in Iterators.take(SobolSeq(n_free(box)), restarts)
        res = optimize(x0) do x
            f(x)
        end
        if res.minimum < best_val
            best_x = res.minimizer
            best_val = res.minimum
        end
    end
    best_x, best_val
end

function minimize(f::Function, condition::Vector{Union{T,Missing}}; restarts=20) where T
    best_val = Inf; best_x = nothing
    for x0 in Iterators.take(SobolSeq(sum(ismissing.(condition))), restarts)
        res = optimize(x0) do x_opt
            x = fillmissing2(condition, x_opt)
            f(x)
        end
        if res.minimum < best_val
            best_x = res.minimizer
            best_val = res.minimum
        end
    end
    best_x, best_val
end


# %% ==================== Load results ====================
# runpath = "tmp/bddm/sobol/v1/"
# subjects = readdir(runpath)
# path = path * subjects[1]
path = "tmp/bddm/sobol/v4/1064"
@unpack box, xs, results, chance = deserialize(path);
ibs_kws = ( ε=.05, tol=0, repeats=50, min_multiplier=1.2)
R = invert(results)
nll = -R.logp .|> fillmissing(-chance * ibs_kws.min_multiplier)
nll_var = R.var .|> fillmissing(20.)

# %% ==================== Empirical minimum ====================

top = partialsortperm(nll, 1:100)
nll[top]
print(xs[top][1])

i = argmin(nll)
emp_x, emp_y = xs[i], nll[i]

# %% ==================== GP minimum ====================

using Distributions

gp = fit_gp(xs, nll, nll_var; gp_mean=chance)

predict_nll(x) = predict_f(gp, reshape(x, n_free(box), 1))[1][1]

function predict_nll(x, quant)
    y, yv = predict_f(gp, reshape(x, n_free(box), 1))
    quantile(Normal(y[1], yv[1]), quant)
end

model_x, model_y = minimize(restarts=100) do x
    predict_nll(x, 0.95)  # robustness -- look for minima with low variance
end

# %% ==================== True NLL ====================

data = filter(d->d.subject == 1064, all_data)
trials = prepare_trials(data; dt=.025)
function true_nll(x)
    m = BDDM(;box(x)...)
    ibs_loglike(m, trials[1:2:end]; ibs_kws...).logp
end

true_nll(model_x)




# %% ==================== Plot the marginals ====================

function plot_marginals(xx, x_best, marginals)
    ps = map(collect(pairs(box.dims)), enumerate(marginals)) do (name, d), (i, x)
        maybelog = :log in d ? (:log,) : ()
        plot(rescale(d, xx), x, xaxis=(string(name), maybelog...))
        hline!([-chance, -chance*1.2], color=:gray, ls=:dash)
        vline!([rescale(d, x_best[i])], c=:red)
    end
    plot(ps..., size=(600, 600))
end

xx = 0:.01:1
x_best = model_x
simple_marginals = map(1:n_free(box)) do i
    x = copy(x_best)
    map(xx) do x_target
        x[i] = x_target
        predict_nll(x)
    end
end




# %% --------
xx = 0:0.05:1

opt_marginals = map(1:n_free(box)) do i
    map(xx) do x_target
        condition = fill(missing, 4)
        # # LBFGS(), autodiff=:forward
        # res = optimize(0, 1) do xi
        #     x = copy(x_best)
        #     x[i] = xi
        #     predict_nll(x, 0.95)
        # end
        # res.minimum
    end
end

figure() do
    plot_marginals(xx, x_best, opt_marginals)
end

# # %% --------
# figure() do
#     ps = map(collect(pairs(box.dims)), enumerate(marginals)) do (name, d), (i, x)
#         maybelog = :log in d ? (:log,) : ()
#         plot(rescale(d, xx), x, xaxis=(string(name), maybelog...))
#     end
#     plot(ps..., size=(600, 600))
# end

