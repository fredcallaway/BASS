using Distributed


@everywhere begin
    include("base.jl")
    include("regressions.jl")
end

version = "2025-05-06"

mkpath("tmp/summary_stats/$version")
mkpath("results/summary_stats/json/$version")

function run_summary(name, data, box)
    study = parse(Int, name[1])
    results = @showprogress name pmap(grid(30, box)) do prm
        if get(prm, :subjective_slope, -1.) == 0.
            subjective_offset = prm.confidence_slope * mean(flatten(data.confidence))
            prm = (;prm..., subjective_offset)
        end
        model = BDDM(;prm...)
        sim = make_sim(model, data; repeats=30)
        accuracy = map(sim) do s
            s.val1 == s.val2 && return missing
            (s.choice == 1) == (s.val1 > s.val2)
        end |> skipmissing |> mean

        (;prm, rt50=median(df.rt), accuracy)
    end
    serialize("tmp/summary_stats/$version/$name", results)
    write("results/summary_stats/json/$version/$name.json", json(results))
end

# %% ==================== Load data ====================

data1 = load_human_data(1)
data2 = load_human_data(2)
avg_conf = mean(flatten(data2.confidence))

# use average confidence from study 2 for study 1 (no confidence judgments)
for d in data1
    d.confidence .= avg_conf
end

# %% ==================== configure search space ====================

box = Box(
    base_precision = 0.,
    attention_factor = 0.,
    confidence_slope = (0.001, 0.1, :log),
    cost = (0.001, 0.1, :log),
)

# %% --------

box1 = let 
    µ, σ = empirical_prior(data1)
    update(box; 
        prior_mean = µ,
        prior_precision = 1 / σ^2,
    )
end

box2 = let 
    µ, σ = empirical_prior(data2)
    update(box; 
        prior_mean = µ,
        prior_precision = 1 / σ^2,
    )
end

# %% ==================== jobs ====================

function run_selected_summary(job_ids)
    jobs = Dict(
        1 => () -> run_summary("2-main", data2, box2),
        2 => () -> run_summary("2-nometa", data2, update(box2, subjective_slope = 0.)),
        # 3 => () -> run_summary("2-biased_mean", data2, update(box2, prior_mean = (0.1µ, 0.9µ))),
        4 => () -> run_summary("2-zero_mean", data2, update(box2, prior_mean = 0.)),
        5 => () -> run_summary("1-main", data1, box1),
        # 6 => () -> run_summary("1-biased_mean", data1, update(box1, prior_mean=(.1µ, .9µ))),
        7 => () -> run_summary("1-zero_mean", data1, update(box1, prior_mean = 0.)),
        8 => () -> run_summary("1-flat_prior", data1, update(box1, prior_precision = 1e-8))
    )

    if isempty(job_ids)
        for job in values(jobs)
            job()
        end
    else
        for job_id in job_ids
            jobs[job_id]()
        end
    end
end

# Call the function with a specific job number or without arguments to run all
run_selected_summary(parse.(Int, ARGS))
