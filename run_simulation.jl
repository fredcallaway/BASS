@everywhere include("model.jl")

vals = -3:1:3
certs = 0.4:0.2:0.8
grid = Iterators.product(vals, vals, certs, certs)
X = pmap(grid) do (v1, v2, c1, c2)
    crt = choice_rt([v1, v2], [c1, c2], [10, 30])
    (v1=v1, v2=v2, c1=c1, c2=c2, crt...)
end

using CSV
X[:] |> CSV.write("results/small.csv")






