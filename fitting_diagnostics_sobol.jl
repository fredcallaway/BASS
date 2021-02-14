using GaussianProcesses
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


function fit_gp(xs, nll, nll_std; gp_mean = mean(nll))
    X = reduce(hcat, xs)
    gp_mean = MeanConst(gp_mean)
    kernel = SEArd(ones(size(X, 1)), -3.)

    log_noise = log.(nll_std)

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
    println("RMSE of perfect prediction: ", √mean(nll_std) |> round2)
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
run_name = "bddm/sobol/v7/"
subjects = parse.(Int, readdir(runpath))
subject = first(subjects)
# path = path * subjects[1]
@unpack box, xs, results, chance, ibs_kws, trials, default = deserialize("tmp/$run_name/$subject");

nll = [-r.logp for r in results]
nll_std = [r.std for r in results] .|> fillmissing(20.)

function true_nll(x; kws...)
    m = BDDM(;box(x)...)
    -ibs_loglike(m, trials[1:2:end]; ibs_kws..., kws...).logp
end


# %% ==================== Empirical minimum ====================

i = argmin(nll)
emp_x, emp_y = xs[i], nll[i]

@show emp_y
@show true_nll(emp_x)

# %% ==================== Plot "samples" ====================
top = partialsortperm(nll, 1:100)
samples = map(pairs(box.dims), xs[top] |> invert) do (name, d), x
    name => x
end |> Dict

params = collect(keys(box.dims))

# %% --------
figure()
    plot_grid(params, params) do vx, vy
        scatter(samples[vx], samples[vy])
        plot!(xlim=(0,1), ylim=(0,1), xlabel=string(vx), ylabel=string(vy))
    end
end

# %% --------
figure() do
    vx = :base_precision
    vy = :confidence_slope
    scatter(samples[vx], samples[vy], smooth=true)
    plot!(xlim=(0,1), ylim=(0,1), xlabel=string(vx), ylabel=string(vy))

end

# %% --------
using DataFrames
using RCall
df = DataFrame(samples)

# %% --------
R"""
round(cor($df), 2)
"""

# %% ==================== GP minimum ====================

using Distributions

# good = findall(nll_std .!= 20)
order = sortperm(nll)
filter!(order) do i
    nll_std[i] < 20
end

# to_fit = [order[1:1000]; order[1001:10:end]]
to_fit = order[1:1000]
gp = fit_gp(xs[to_fit], nll[to_fit], nll_std[to_fit]; gp_mean=-ibs_kws.min_multiplier * chance)

predict_nll(x) = predict_f(gp, reshape(x, n_free(box), 1))[1][1]

function predict_nll(x, quant)
    y, yv = predict_f(gp, reshape(x, n_free(box), 1))
    quantile(Normal(y[1], yv[1]), quant)
end

function predict_nll_meanstd(x)
    y, yv = predict_f(gp, reshape(x, n_free(box), 1))
    y[1], √yv[1]
end

model_x, model_y = minimize(restarts=100) do x
    predict_nll(x, 0.9)  # robustness -- look for minima with low variance
end

@show model_y
true_model_y = @show true_nll(model_x)
true_emp_y = @show true_nll(emp_x)


best_x = if true_model_y < true_emp_y
    println("Using model's best point")
    model_x
else
    println("Using empirical best point")
    emp_x
end

# %% --------
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
map(1:10000) do i
    t = SimTrial()
    sim = simulate(m, t)
    pt1, pt2 = sim.presentation_times .* t.dt
    val1, val2 = t.value
    conf1, conf2 = t.confidence
    m1, m2 = mean.(t.presentation_times)
    order = m1 > m2 ? :longfirst : :shortfirst
    (;subject, val1, val2, conf1, conf2, pt1, pt2, order, sim.choice)
end |> CSV.write("results/v7-1064-unyoked.csv")



# %% ==================== Plot the marginals ====================

function plot_marginals(xx, best_x, marginals)
    ps = map(collect(pairs(box.dims)), enumerate(marginals)) do (name, d), (i, x)
        maybelog = :log in d ? (:log,) : ()
        plot(rescale(d, xx), x, xaxis=(string(name), maybelog...))
        hline!([-chance, -chance*1.2], color=:gray, ls=:dash)
        vline!([rescale(d, best_x[i])], c=:red)
    end
    plot(ps..., size=(600, 600))
end

function plot_marginals_meanstd(xx, best_x, marginals)
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
    plot_marginals_meanstd(xx, best_x, simple_marginals)
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

