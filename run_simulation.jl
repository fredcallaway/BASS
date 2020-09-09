using Distributed
using CSV
using DataFrames
using ProgressMeter
using Random


@everywhere begin
    include("model.jl")
    include("dc.jl")
end
include("utils.jl")

@everywhere function run_sim(m::BDDM, t::Trial)
    pol = DirectedCognition(m)
    sim = simulate(m, pol; t=t)
    pt1, pt2 = sim.presentation_times
    (namedtuple(m)..., namedtuple(t)..., pt1=pt1, pt2=pt2, choice=sim.choice, rt=sim.rt)
end

function run_sims(bddms::Vector{BDDM}, N::Int)
    @showprogress @distributed vcat for i in 1:N
        t = Trial(randn(2), .5 .+ 2rand(2), shuffle!([Normal(10, 2.5), Normal(25, 5)]))
        map(bddms) do m
            run_sim(m, t)
        end
    end
end

function run_many(name, args; skip_cols=[], force=false)
    file = "results/$name.csv"
    if isfile(file) && !force
        error("$file already exists.")
    end
    println("Creating $file")

    bddms = map_product(BDDM; args...)[:]
    df = DataFrame(run_sims(bddms, Int(1e5)))
    
    drop = setdiff(fieldnames(BDDM), setdiff(keys(args), skip_cols))
    select!(df, Not(intersect(propertynames(df), drop)))

    x = Between(:val1, :conf2)
    df[!, x] = round.(df[!, x]; sigdigits=4)

    CSV.write(file, df)
    println("Wrote $(size(df, 1)) rows to $file")
    df
end

# %% --------

args = Dict(
    :cost => [4e-4],
    :risk_aversion => [0., 16e-3, 64e-3],
    :over_confidence_slope => [1.1, 1.5, 2],
)
run_many("sep9-basic.csv", args; skip_cols=[:cost])


args = Dict(
    :cost => [4e-4],
    :risk_aversion => [0., 16e-3, 64e-3],
    :over_confidence_slope => [1.1, 1.5, 2],
    :prior_mean => -1.7
)
run_many("sep9-stupid_prior.csv", args; skip_cols=[:cost])


args = Dict(
    :cost => [4e-4],
    :risk_aversion => [0., 16e-3, 64e-3],
    :over_confidence_slope => [0],
    :over_confidence_intercept => collect(range(0.5, 2.5, length=5))[2:end-1],
)
run_many("sep9-stupid_confidence.csv", args; skip_cols=[:cost])
