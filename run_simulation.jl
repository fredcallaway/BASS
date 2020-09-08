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

function run_many(name args)
    bddms = map_product(BDDM; args...)[:]
    df = DataFrame(run_sims(bddms, Int(1e5)))
    d = df[!, 4:end-1]
    d[!, 1:4] = round.(d[!, 1:4]; digits=4)
    CSV.write("results/$name.csv", d)
    println("Wrote $(size(d, 1)) rows to results/$name.csv")
    d
end

# %% --------

args = Dict(
    :cost => [4e-4],
    :risk_aversion => [0., 16e-3, 64e-3],
    :over_confidence => [1.1, 1.5, 2, 4],
)

run_many("results/sep8-replicate-jul24-D.csv", args)
