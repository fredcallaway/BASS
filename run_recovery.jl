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
    base_precision = (.01, 1, :log),
    attention_factor = (0, 1),
    cost = (.01, .03, :log),
    confidence_slope = (0, 0.25),
    prior_mean = (-2, 0),
)

n_subj = 20
dt = .1
xs = Iterators.take(SobolSeq(n_free(sim_box)), n_subj) |> collect

# n_trial = 5000
# trial_sets = map([1]) do data
#     [SimTrial(; dt) for i in 1:n_trial]
# end |> Iterators.cycle |> (x -> Iterators.take(x, length(xs)))

all_data = load_human_data()
trial_sets = map(collect(group(d->d.subject, all_data))) do data
    SimTrial.(prepare_trials(Table(data), dt=.1))
end |> Iterators.cycle  |> (x -> Iterators.take(x, length(xs)))

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

sim_data = @showprogress "Simulating " map(enumerate(xs), trial_sets) do (subject, x), trials
    m = BDDM(;sim_box(x)...)
    simulate_dataset(m, trials; subject)
end

# %% ==================== Check summary statistics ====================

function summarize(t::NamedTuple)
    vd = t.value[1] - t.value[2]
    choose_best = 
        vd < 0 ? t.choice == 2 :
        vd > 0 ? t.choice == 1 :
        missing
    (
        ;t.rt, choose_best,
        choose_first = t.choice .== 1,
    )
end

function summarize(ts::AbstractVector{<:NamedTuple})
    map(invert(map(summarize, ts))) do x
        mean(skipmissing(x))
    end
end

human_sum = summarize(all_data)
model_sum = map(summarize, sim_data) |> Table

M = Table(Table(subject=eachindex(sim_data)), Table(sim_box.(xs)), model_sum)
M |> CSV.write("tmp/qualitative.csv")
Table([human_sum]) |> CSV.write("tmp/qualitative_human.csv")
convert(Dict, map(quantile, columns(model_sum)))

# %% ==================== Sobol search ====================
version = "recovery/v6"
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
    @show g.gp.kernel

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




