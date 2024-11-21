include("base.jl")
using DataFrames, CSV

version = "2024-11-20"
params = deserialize("tmp/bddm/grid/recovery/$version/params")

mkpath("results/recovery/$version")


path = "tmp/bddm/grid/recovery/$version/"
flatmap(eachindex(params)) do param_id
    flatmap(readdir("$path/$param_id")) do subject
        x = deserialize("$path/$param_id/$subject")
        map(x.candidates[:], x.results[:]) do c, r
            (; param_id, subject, c.base_precision, c.attention_factor, c.cost, r.logp, r.std, r.converged)
        end
    end
end |> CSV.write("results/recovery/$version/likelihoods.csv", writeheader=true)

map(eachindex(params)) do param_id
    (;param_id, params[param_id]...)
end |> CSV.write("results/recovery/$version/generating_params.csv", writeheader=true)

