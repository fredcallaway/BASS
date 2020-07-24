using Distributed
using CSV
using DataFrames
using ProgressMeter
using Random

@everywhere begin
    include("model.jl")
    include("dc.jl")
end

@everywhere function run_sim(t::Trial, cost, risk)
    m = BDDM(base_precision=.01, cost=cost, risk_aversion=risk)
    pol = DirectedCognition(m)
    sim = simulate(m, pol; t=t)
    pt1, pt2 = sim.presentation_times
    (namedtuple(m)..., namedtuple(t)..., pt1=pt1, pt2=pt2, choice=sim.choice, rt=sim.rt)
end

# %% ==================== Jul 24 ====================

costs = [4e-4]
risks = [2e-3, 4e-3, 16e-3, 32e-3]
N = Int(1e5)
args = Iterators.product(costs, risks, 1:N)
length(args)
results = @showprogress pmap(args, batch_size=1000) do (cost, risk, i)
    x = 0.5
    t = Trial(randn(2), .5 .+ 2rand(2), shuffle!([Normal(20x, 5x), Normal(50x, 10x)]))
    run_sim(t, cost, risk)
end
df = DataFrame(results)

d = df[!, 4:end-1]
d[!, 1:4] = round.(d[!, 1:4]; digits=4)
d |> CSV.write("results/jul24-C.csv")

# %% ==================== Jun 28 ====================


costs = [4e-4]
risks = [0]
N = Int(1e6)
args = Iterators.product(costs, risks, 1:N)
@time df = pmap(args, batch_size=1000) do (cost, risk, i)
    p = Problem(randn(2), .5 .+ 2rand(2), rand(10:30, 2))
    run_sim(p, cost, risk)
end |> DataFrame

# %% --------
d = df[!, 5:end-1]
d[!, 1:4] = round.(d[!, 1:4]; digits=3)
d |> CSV.write("results/jun29.csv")



# %% ==================== Jun 24 ====================

costs = [4e-4]
risks = [0, 1e-4, 4e-4]
vals = [randn(2) for i in 1:100]
ptimes = [10, 30, 50]
certs = [0.5, 1, 2]
grid = Iterators.product(costs, risks, ptimes, vals, certs, certs, ptimes, ptimes)
println(length(grid))
@time X = pmap(grid) do (cost, risk, pt2, (v1, v2), c1, c2, pt1, pt2)
    p = Problem([v1, v2], [c1, c2], [pt1, pt2])
    crt = choice_rt(p, cost, risk; n_sim=1000)
    (cost=cost, risk=risk, val1=v1, val2=v2, conf1=c1, conf2=c2, pt1=pt1, pt2=pt2, crt...)
end;

using CSV
X[:] |> CSV.write("results/jun24-2.csv")



