@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
    using Serialization
end
using ProgressMeter
using Sobol
using SplitApplyCombine
using CSV

include("sobol_search.jl")
mkpath("tmp/recovery")

# %% ==================== Simulate ====================
sim_box = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1),
    cost = (.01, .1),
    # confidence_slope = (0, 0.25),
    prior_mean = (-2, 0),
)

n_subj = 20
dt = .1
xs = Iterators.take(SobolSeq(n_free(sim_box)), n_subj) |> collect

# n_trial = 5000
# trial_sets = map([1]) do data
#     [SimTrial(; dt) for i in 1:n_trial]
# end |> Iterators.cycle |> (x -> Iterators.take(x, length(xs)))

all_data = load_human_data(1)
trial_sets = map(collect(group(d->d.subject, all_data))) do data
    SimTrial.(prepare_trials(Table(data), dt=.1))
end |> Iterators.cycle |> (x -> Iterators.take(x, length(xs)))

function simulate_dataset(m, trials; subject=0)
    map(trials) do t
        sim = simulate(m, t; save_presentation=true)
        presentation_duration = t.dt .* sim.presentation_durations
        m1, m2 = mean.(t.presentation_distributions)
        order = m1 > m2 ? :longfirst : :shortfirst
        rt = sim.time_step .* t.dt
        (;subject, t.value, t.confidence, presentation_duration, order, sim.choice, rt)
    end
end


data1 = load_human_data(1)
µ, σ = empirical_prior(data1)
m1_main = BDDM(
    base_precision = .05,
    attention_factor = 0.8,
    cost = .06,
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)

sim_data = Table(simulate_dataset(m1_main, prepare_trials(data1)))

# sim_data = @showprogress "Simulating " map(enumerate(xs), trial_sets) do (subject, x), trials
#     m = BDDM(;sim_box(x)...)
#     simulate_dataset(m, trials; subject)
# end

# %% ==================== Sobol search ====================

box = Box(
    base_precision = (.01, .1),
    attention_factor = (0, 1.5),
    cost = (.01, .1),
    prior_mean = (0, 2µ),
    prior_precision = 1 / σ^2,
)


version = "recovery/apr10"

res = sobol_search(box, 10, sim_data, tol=20, min_multiplier=1.)

best = argmax(invert(res.results).logp)

res.models[best]


run_sobol_ind(BDDM, version, sim_box, 3000, repeats=50, dt=.1, tol=0; data=flatten(sim_data))


# %% ==================== Process ====================
@everywhere include("gp_likelihood.jl")
@everywhere begin
    xs = $xs
    version = $version
end
subjects = readdir("tmp/bddm/sobol/$version/")
# @showprogress "Processing sobol results" pmap(subjects) do subject
#     process_sobol_result(BDDM, "v7", subject)
# end

@showprogress pmap(subjects; on_error=identity) do subject
    # isfile("figs/$version/$subject.png") && return

    true_x = xs[parse(Int, subject)]
    repeats = 100
    opt_points = 1000
    verbose = true
    sr = SobolResult(BDDM, version, subject);
    g = GPSL(sr; opt_points)

    model_x, model_nll_hat, model_nll_true = model_minimum(g; repeats)
    

    x0 = true_x
    xx_true = 0:.1:1
    true_marginals = map(1:length(x0)) do i
        x = copy(x0)
        y, y_std = map(xx_true) do x_target
            x[i] = x_target
            m = sr.model(;sr.box(x)...)
            res = ibs_loglike(m, sr.trials[1:2:end]; sr.ibs_kws..., repeats)
            (res.logp, res.std)
        end |> invert
    end
    xx_pred = 0:.01:1
    predicted_marginals = map(1:length(x0)) do i
        x = copy(x0)
        y, y_std = map(xx_pred) do x_target
            x[i] = x_target
            predict_meanstd(g, x)
        end |> invert
    end

    y_true, y_true_std = true_nll_meanstd(sr, true_x; repeats=1000)
    # %% --------

    figure("$version/$subject") do
        chance = chance_loglike(sr)
        plots = map(enumerate(pairs(sr.box.dims))) do (i, (name, d))
            maybelog = :log in d ? (:log,) : ()
            y, ystd = true_marginals[i]
            ystd = replace(ystd, missing => NaN)
            ystd .*= 2
            scatter(rescale(d, xx_true), -y, color=:black, markersize=3, err=ystd)

            y, ystd = predicted_marginals[i]
            try
                plot!(rescale(d, xx_pred), y, xaxis=(string(name), maybelog...), ribbon=ystd, color=1)
            catch
                println("error 1") 
            end

            try
                hline!([y_true], ribbon=2y_true_std, color=:black, fillalpha=.2)
            catch
                println("error 2") 
            end

            hline!([-chance], color=:gray, ls=:dash)
            vline!([rescale(d, x0[i])], c=:red)
            plot!(ylim=(-Inf, -chance*1.01))
        end
        plot(plots..., size=(600, 600))
    end
end




