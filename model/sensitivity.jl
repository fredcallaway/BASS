using Distributed

# println("Building RCall")
# using Pkg
# Pkg.build("RCall")

@everywhere begin
    include("base.jl")
    include("regressions.jl")
end

version = "2025-04-11"

mkpath("tmp/sensitivity/$version")
mkpath("results/sensitivity-json/$version")

using ProgressMeter
using Sobol
function sobol(n::Int, box::Box)
    seq = SobolSeq(length(box))
    skip(seq, n)
    [box(Sobol.next!(seq)) for i in 1:n]
end

function run_sensitivity(name, data, box)
    study = parse(Int, name[1])
    results = @showprogress name pmap(grid(11, box)) do prm
        if isnan(get(prm, :subjective_offset, 0.)) # NaN is a flag for 2-nometa
            subjective_offset = prm.confidence_slope * mean(flatten(data.confidence))
            prm = (;prm..., subjective_offset)
        end
        model = BDDM(;prm...)
        df = DataFrame(make_sim(model, data; repeats=30))

        (;prm, fit_regressions(df; study)...)
    end
    serialize("tmp/sensitivity/$version/$name", results)
    write("results/sensitivity-json/$version/$name.json", json(results))
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
    confidence_slope = (1e-5, 0.02),
    attention_factor = 0.,
    cost = (0.01, 0.1),

)

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

function run_selected_sensitivity(job_ids)
    jobs = Dict(
        1 => () -> run_sensitivity("2-main", data2, box2),
        2 => () -> run_sensitivity("2-nometa", data2, update(box2, subjective_slope = NaN)),
        # 3 => () -> run_sensitivity("2-biased_mean", data2, update(box2, prior_mean = (0.1µ, 0.9µ))),
        4 => () -> run_sensitivity("2-zero_mean", data2, update(box2, prior_mean = 0.)),
        5 => () -> run_sensitivity("1-main", data1, box1),
        # 6 => () -> run_sensitivity("1-biased_mean", data1, update(box1, prior_mean=(.1µ, .9µ))),
        7 => () -> run_sensitivity("1-zero_mean", data1, update(box1, prior_mean = 0.)),
        8 => () -> run_sensitivity("1-flat_prior", data1, update(box1, prior_precision = 1e-8))
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
run_selected_sensitivity(parse.(Int, ARGS))
