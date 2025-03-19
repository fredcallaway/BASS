include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("box.jl")
using Serialization
using CSV

# %% --------



function empirical_prior(data; α=1)
    µ, σ = juxt(mean, std)(flatten(data.value))
    α * µ, σ
end

function make_frame(data)
   map(data) do d
       val1, val2 = d.value
       conf1, conf2 = d.confidence
       pt1 = round(sum(d.presentation_duration[1:2:end]); digits=3)
       pt2 = round(sum(d.presentation_duration[2:2:end]); digits=3)
       (;d.subject, val1, val2, conf1, conf2, pt1, pt2, d.choice, d.rt, d.order,
        initpresdur1=d.presentation_duration[1], initpresdur2=get(d.presentation_duration, 2, missing))
   end |> Table
end

function simulate_dataset(m, trials; ndt=0)
    map(trials) do t
        sim = simulate(m, SimTrial(t; dt=.025); save_presentation=true)
        presentation_duration = t.dt .* sim.presentation_durations
        m1, m2 = mean.(t.presentation_distributions)
        order = m1 > m2 ? :longfirst : :shortfirst
        rt = sim.time_step .* t.dt + ndt
        (;t.subject, t.value, t.confidence, presentation_duration, order, sim.choice, rt)
    end
end

function make_sim(model, data; normalize_value=false, repeats=30)
    trials = repeat(prepare_trials(Table(data); normalize_value), repeats);
    df = make_frame(simulate_dataset(model, trials))
    if normalize_value
        # unnomrmalize it
        val_µ, val_σ = empirical_prior(data)
        @. df.val1 = round(df.val1 * val_σ + val_µ; digits=2)
        @. df.val2 = round(df.val2 * val_σ + val_µ; digits=2)
    end
    df
end

function write_sim(model, data, version, name; normalize_value=false, repeats=30)
    mkpath("results/$version")
    df = make_sim(model, data; normalize_value, repeats)
    fn = "results/$version/$name.csv"
    df |> CSV.write(fn)
    println("wrote $fn")
    df
end
