using Distributed
@everywhere begin
    include("base.jl")
    include("regressions.jl")
end

mkpath("results/sensitivity")
mkpath("results/sensitivity-json")

using ProgressMeter
using Sobol
function sobol(n::Int, box::Box)
    seq = SobolSeq(length(box))
    skip(seq, n)
    [box(Sobol.next!(seq)) for i in 1:n]
end

function run_sensitivity(name, data, box; N=1000)
    study = parse(Int, name[1])
    results = @showprogress name pmap(sobol(N, box)) do prm
        if ismissing(get(prm, :subjective_offset, nothing))
            subjective_offset = prm.confidence_slope * mean(flatten(data.confidence))
            prm = (;prm..., subjective_offset)
        end
        model = BDDM(;prm...)
        df = DataFrame(make_sim(model, data; repeats=30))



        (;prm, fit_regressions(df; study)...)
    end
    serialize("results/sensitivity/$name", results)
    write("results/sensitivity-json/$name.json", json(results))
end


# %% ==================== study 1 ====================


# data1 = load_human_data(1)
# µ, σ = empirical_prior(data1)
# box1 = Box(
#     base_precision = (.01, .1),
#     attention_factor = (0, 1.),
#     cost = (.01, .1),
#     prior_mean = µ,
#     prior_precision = 1 / σ^2,
# )

# run_sensitivity("1-main", data1, box1)
# run_sensitivity("1-biased_mean", data1, update(box1, prior_mean=(µ/2, µ)))
# run_sensitivity("1-zero_mean", data1, update(box1, prior_mean = 0.))
# run_sensitivity("1-flat_prior", data1, update(box1, prior_precision = 1e-8))

# %% ==================== study 2 ====================

data2 = load_human_data(2)
µ, σ = empirical_prior(data2)

box2 = Box(
    base_precision = (.001, .1, :log),
    confidence_slope = (.001, .02),
    attention_factor = (0, 1.),
    cost = (.01, .1),
    prior_mean = µ,
    prior_precision = 1 / σ^2,
    subjective_offset = 0.,
    subjective_slope = 1.,
)

run_sensitivity("2-main", data2, box2)
run_sensitivity("2-biased_mean", data2, update(box2, prior_mean = (0.1µ, 0.9µ)))
run_sensitivity("2-zero_mean", data2, update(box2, prior_mean = 0.))
run_sensitivity("2-nometa", data2, update(box2, subjective_slope = 0, subjective_offset = missing))
