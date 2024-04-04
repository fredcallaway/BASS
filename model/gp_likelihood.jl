using GaussianProcesses
using Distributions
using Parameters
using Serialization
using SplitApplyCombine
using Statistics
using Optim
using Sobol
using CSV

include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("likelihood.jl")
include("ibs.jl")
include("box.jl")
include("figure.jl")
include("minimize.jl")

# %% ==================== Load results ====================

struct SobolResult
    model::DataType
    version::String
    subject::String
    trials::Vector{HumanTrial}
    ibs_kws::NamedTuple
    box::Box
    X::Matrix{Float64}
    nll::Vector{Float64}
    nll_std::Vector{Float64}
end

function SobolResult(model, version, subject)
    path = "tmp/$(lowercase(string(model)))/sobol/$version/$subject"
    @unpack box, xs, results, chance, ibs_kws, trials = deserialize(path);

    X = combinedims(xs)
    nll = [-r.logp for r in results]
    nll_std = [r.std for r in results] .|> fillmissing(20.)

    SobolResult(model, version, subject, trials, ibs_kws, box, X, nll, nll_std)
end

identifier(sr::SobolResult) = join([lowercase(string(sr.model)), sr.version, sr.subject], "-")
Base.show(io::IO, sr::SobolResult) = print(io, "SobolResult($(identifier(sr)))")


function true_nll_meanstd(sr::SobolResult, x; repeats=100, kws...)
    m = sr.model(;sr.box(x)...)
    res = ibs_loglike(m, sr.trials; sr.ibs_kws..., repeats, kws...)
    (-res.logp, res.std)
end

function true_nll(sr::SobolResult, x; kws...)
    true_nll_meanstd(sr, x; kws...)[1]
end

function empirical_minimum(sr::SobolResult; repeats=100, verbose=true)
    i = argmin(sr.nll)
    x = sr.X[:, i]
    recomputed = round1(true_nll(sr, x; repeats))
    x, sr.nll[i], recomputed
end

# %% ==================== GP Surrogate ====================

struct GPSL # gaussian process surrogate likelihood
    sr::SobolResult
    gp::GPE
end

function fit_gp(X, nll, nll_std; gp_mean = mean(nll), opt_points=1000)
    gp_mean = MeanConst(gp_mean)
    # this initialization seems to work well for some reason...
    # kernel = SEArd(randn(size(X, 1)), log(maximum(nll) - minimum(nll)))
    kernel = SEArd(-2. * ones(size(X, 1)), -1.)
    log_noise = log.(nll_std)
    opt_idx = 1:min(size(X, 2), opt_points)
    gp = GPE(X[:, opt_idx], nll[opt_idx], gp_mean, kernel, log_noise[opt_idx])
    optimize!(gp, domean=false, noise=false, Optim.Options(iterations=100, f_tol=.001))

    yhat, yvar = predict_f(gp, X[:, opt_idx[end]:end])
    ytrue = nll[opt_idx[end]:end]
    √ mean((yhat .- ytrue).^2)

    GPE(X, nll, gp_mean, gp.kernel, log_noise)
end

# %% --------

function cross_validate_gp(X, nll, nll_std; verbose=false, kws...)
    train = eachindex(nll)[1:2:end]
    test = eachindex(nll)[2:2:end]
    gp = fit_gp(X[:, train], nll[train], nll_std[train]; kws...)
    yhat, yvar = predict_f(gp, X[:, test])
    rmse(preds) = √ mean((preds .- nll[test]).^2)

    if verbose
        println("RMSE on held out data: ", rmse(yhat) |> round2)
        println("RMSE of mean prediction: ", rmse(mean(nll[train])) |> round2)
        println("RMSE of perfect prediction: ", mean(nll_std) |> round2)
        println("RMSE scaled by prediction variance: ", 
            √mean(((yhat .- nll[test]) ./ .√yvar).^2) |> round2)
    end
    return rmse(yhat)
end

function predict_meanstd(g::GPSL, x)
    f, fv = predict_f(g.gp, reshape(x, g.gp.dim, 1))
    f[1], √fv[1]
end

function predict_quantile(g::GPSL, x, q)
    f, σ = predict_meanstd(g, x)
    quantile(Normal(f, σ), q)
end

function find_minimum(g::GPSL; restarts=10)
    minimize(g.gp.dim; restarts, method=BFGS(), autodiff=:forward) do x
        predict_quantile(g, x, 0.9)  # robustness -- look for minima with low variance
    end
end

 function model_minimum(g::GPSL; repeats=100)
    model_x, model_nll = find_minimum(g)
    (model_x, model_nll, true_nll(g.sr, model_x; repeats))
end

function estimate_error(g::GPSL, x; q=0.5)
    true_nll(g.sr, x) - predict_quantile(g, x, q)
end

chance_loglike(sr::SobolResult) = chance_loglike(sr.trials; sr.ibs_kws.tol)
function GPSL(sr::SobolResult; kws...)
    gp_mean = -sr.ibs_kws.min_multiplier * chance_loglike(sr)
    converged = sr.nll_std .≠ 20
    X = sr.X[:, converged]
    nll = sr.nll[converged]
    nll_std = sr.nll_std[converged]
    gp = fit_gp(X, nll, nll_std; gp_mean, kws...)
    GPSL(sr, gp)
end


# %% ==================== Plots ====================

function plot_marginals(g::GPSL, x0)
    figure("$(identifier(g.sr))-marginals") do
        xx = 0:.01:1
        chance = chance_loglike(g.sr)
        plots = map(enumerate(free(g.sr.box))) do (i, name)
            d = g.sr.box[name]
            x = copy(x0)
            y, ystd = map(xx) do x_target
                x[i] = x_target
                predict_meanstd(g, x)
            end |> invert
            maybelog = :log in d ? (:log,) : ()
            plot(rescale(d, xx), y, xaxis=(string(name), maybelog...), ribbon=ystd)
            hline!([-chance, -chance*1.2], color=:gray, ls=:dash)
            vline!([rescale(d, x0[i])], c=:red)

        end
        plot(plots..., size=(600, 600))
    end
end

# %% ==================== Main ====================

#=
model = BDDM
version = "v12"
repeats = 1
opt_points = 1000
verbose = true
subject = "group"
=#

function process_sobol_result(model, version, subject; repeats=100, opt_points=1000, verbose=false)
    sr = SobolResult(model, version, subject);
    emp_x, emp_nll_hat, emp_nll_true = empirical_minimum(sr; repeats)

    g = GPSL(sr; opt_points)
    model_x, model_nll_hat, model_nll_true = model_minimum(g; repeats)

    if verbose
        println("Empirical minimum:")
        println("  x              = ", round3.(emp_x))
        println("  observed nll   = ", round1(emp_nll_hat))
        println("  recomputed nll = ", emp_nll_true)
        println()
        println("Model minimum:")
        println("  x             = ", round3.(model_x))
        println("  predicted nll = ", round3.(model_nll_hat))
        println("  computed nll  = ", model_nll_true)
    end

    best_x = if (repeats > 10) && model_nll_true < emp_nll_true
        println("Using model's best point")
        model_x
    else
        println("Using empirical best point")
        emp_x
    end
    plot_marginals(g, best_x)

    prm = sr.box(best_x)
    res = (; emp_x, emp_nll_hat, emp_nll_true, model_x, model_nll_hat, model_nll_true, best_x, prm, sr.trials[1].dt)
    serialize("results/$(identifier(sr))-mle", res)
    return res
end
