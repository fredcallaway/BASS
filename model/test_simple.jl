include("model.jl")
include("dc.jl")
include("utils.jl")
include("data.jl")

using Test

µ0 = -1
λ0 = .1
λ_obs = .4
v = 2

using Test
@test d.µ ≈ mean(µ1s) atol=.01
@test d.σ ≈ std(µ1s) atol=.01