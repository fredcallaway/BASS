using GaussianProcesses
using Distributions
using Parameters
using Serialization
using SplitApplyCombine
using Statistics
using Optim
using Sobol
using CSV
using Strs

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

struct SobolResult3
    model::DataType
    version::String
    subject::Int
    trials::Vector{HumanTrial}
    ibs_kws::NamedTuple
    box::Box
    X::Matrix{Float64}
    nll::Vector{Float64}
    nll_std::Vector{Float64}
end


function SobolResult3(model, version, subject)
    path = "tmp/$(lowercase(string(model)))/sobol/$version/$subject"
    @unpack box, xs, results, chance, ibs_kws, trials, default = deserialize(path);

    X = combinedims(xs)
    nll = [-r.logp for r in results]
    nll_std = [r.std for r in results] .|> fillmissing(20.)

    SobolResult3(model, version, subject, trials, ibs_kws, box, X, nll, nll_std)
end

function true_nll(sr::SobolResult3, x; kws...)
    m = sr.model(;sr.box(x)...)
    -ibs_loglike(m, sr.trials[1:2:end]; sr.ibs_kws..., kws...).logp
end

function empirical_minimum(sr::SobolResult3; verbose=true)
    i = argmin(sr.nll)
    x = sr.X[:, i]
    if verbose
        println("Empirical minimum:")
        println("  x              = ", round3.(x))
        println("  observed nll   = ", round1(sr.nll[i]))
        println("  recomputed nll = ", round1(true_nll(sr, x)))
    end
end

# %% ==================== GP Surrogate ====================

struct GPSL2 # gaussian process surrogate likelihood
    sr::SobolResult3
    gp::GPE
end

function fit_gp(X, nll, nll_std; gp_mean = mean(nll))
    gp_mean = MeanConst(gp_mean)
    kernel = SEArd(ones(size(X, 1)), -3.)
    log_noise = log.(nll_std)
    gp = GPE(X, nll, gp_mean, kernel, log_noise)
    optimize!(gp, domean=false, noise=false)
    gp
end

function cross_validate_gp(X, nll, nll_std; verbose=true, kws...)
    train = eachindex(nll)[1:2:end]
    test = eachindex(nll)[2:2:end]
    gp = fit_gp(X[:, train], nll[train], nll_std[train]; kws...)
    yhat, yvar = predict_f(gp, X[:, test])
    rmse(preds) = √ mean((preds .- nll[test]).^2)
    if verbose
        println("RMSE on held out data: ", rmse(yhat) |> round2)
        println("RMSE of mean prediction: ", rmse(mean(nll[train])) |> round2)
        println("RMSE of perfect prediction: ", √mean(nll_std) |> round2)
        println("RMSE scaled by prediction variance: ", 
            mean(((yhat .- nll[test]) ./ .√yvar).^2) |> round2)
    end
    return rmse(yhat)
end

# good = findall(nll_std .!= 20)
order = sortperm(nll)
filter!(order) do i
    nll_std[i] < 20
end

# to_fit = [order[1:1000]; order[1001:10:end]]
to_fit = order[1:1000]
gp = fit_gp(xs[to_fit], nll[to_fit], nll_std[to_fit]; gp_mean=-ibs_kws.min_multiplier * chance)


function predict_meanstd(g::GPSL2, x)
    f, fv = predict_f(g.gp, reshape(x, n_free(box), 1))
    f[1], √fv[1]
end

function predict_quantile(g::GPSL2, x, q)
    f, σ = predict_meanstd(x)
    quantile(Normal(f, σ), q)
end

function find_minimum(g::GPSL2)
    model_x, model_y = minimize(restarts=100) do x
        predict_quantile(g, x, 0.9)  # robustness -- look for minima with low variance
    end
end

function estimate_error(g::GPLS2, x; q=0.5)
    true_nll(g.sr, x) - predict_quantile(g, x, q)
end

true_model_y = true_nll(model_x)
true_emp_y = @show true_nll(emp_x)
@show model_y


best_x = if true_model_y < true_emp_y
    println("Using model's best point")
    model_x
else
    println("Using empirical best point")
    emp_x
end

# %% ====================  ====================

run_name = "bddm/sobol/v7/"
subjects = parse.(Int, readdir(runpath))
sr = @time SobolResult3(BDDM, "v7", 1064);
empirical_minimum(sr)



# %% ==================== Plot the marginals ====================

function plot_marginals(xx, best_x, marginals)
    ps = map(collect(pairs(box.dims)), enumerate(marginals)) do (name, d), (i, x)
        y, ystd = invert(x)
        maybelog = :log in d ? (:log,) : ()
        plot(rescale(d, xx), y, xaxis=(string(name), maybelog...), ribbon=ystd)
        hline!([-chance, -chance*1.2], color=:gray, ls=:dash)
        vline!([rescale(d, best_x[i])], c=:red)
    end
    plot(ps..., size=(600, 600))
end

xx = 0:.01:1
simple_marginals = map(1:n_free(box)) do i
    x = copy(best_x)
    map(xx) do x_target
        x[i] = x_target
        predict_nll_meanstd(x)
    end
end


figpath = "figs/" * replace(replace(run_name, "sobol/" => ""), "/" => "-") * subject

figure("$figpath-simple_marginal") do
    plot_marginals(xx, best_x, simple_marginals)
end

# %% --------
xx = 0:0.05:1

opt_marginals = map(1:n_free(box)) do i
    @showprogress map(xx) do x_target
        condition = Vector{Union{Float64,Missing}}(fill(missing, n_free(box), ))
        condition[i] = x_target
        minimize(condition) do x
            predict_nll(x, 0.95)
        end
    end
end

# %% --------
y = map(opt_marginals) do xs
    [x[2] for x in xs]
end
# %% --------
figure() do
    plot_marginals(xx, best_x, y)
end

# # %% --------
# figure() do
#     ps = map(collect(pairs(box.dims)), enumerate(marginals)) do (name, d), (i, x)
#         maybelog = :log in d ? (:log,) : ()
#         plot(rescale(d, xx), x, xaxis=(string(name), maybelog...))
#     end
#     plot(ps..., size=(600, 600))
# end


# %% ==================== Simulate ====================


m = BDDM(;box(best_x)...)


mapmany(trials[1:end]) do t
    map(1:1000) do i
        sim = simulate(m, t)
        pt1, pt2 = sim.presentation_times .* t.dt
        val1, val2 = t.value
        conf1, conf2 = t.confidence
        m1, m2 = mean.(t.presentation_times)
        order = m1 > m2 ? :longfirst : :shortfirst
        (;t.subject, val1, val2, conf1, conf2, pt1, pt2, order, sim.choice)
    end
end |> CSV.write("results/v7-1064.csv")

# %% --------

function simulate(m; dt)
    map(1:10000) do i
        t = SimTrial()
        sim = simulate(m, t)
        pt1, pt2 = sim.presentation_times .* t.dt
        val1, val2 = t.value
        conf1, conf2 = t.confidence
        m1, m2 = mean.(t.presentation_times)
        order = m1 > m2 ? :longfirst : :shortfirst
        (;subject, val1, val2, conf1, conf2, pt1, pt2, order, sim.choice)
    end
end
simulate(m; trials[1].dt) |> CSV.write("results/v7-1064-unyoked.csv")

