@everywhere include("model.jl")

@everywhere function choice_rt(t::Trial; n_sim=10000, kws...)
    m = BDDM(base_precision=.01, cost=1e-4)
    pol = DirectedCognition(m)
    choice, rt = n_sim \ mapreduce(+, 1:n_sim) do i
        sim = simulate(m, pol; t=t)
        [sim.choice - 1, sim.rt]
    end
    (choice=choice, rt=rt)
end


vals = -3:1:3
certs = 0.4:0.2:0.8
grid = Iterators.product(vals, vals, certs, certs)
@time X = pmap(grid) do (v1, v2, c1, c2)
    t = Trial([v1, v2], [c1, c2], [10, 30])
    crt = choice_rt(t)
    (v1=v1, v2=v2, c1=c1, c2=c2, crt...)
end;

using CSV
X[:] |> CSV.write("results/small.csv")






