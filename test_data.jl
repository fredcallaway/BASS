include("utils.jl")
include("model.jl")
include("data.jl")

using Test


all_data = load_human_data()
all_trials = prepare_trials(all_data)

@testset "discretize_presentation_times" begin
    dt = .025
    errs = map(all_data) do d
        target = sum(d.presentation_duration)
        after_discretization = sum(discretize_presentation_times(d.presentation_duration, dt)) * dt
        after_discretization - target
    end
    @test mean(errs) ≈ 0  atol=.001
    @test maximum(abs.(errs)) < dt/2
end

foreach(all_trials) do t
    @assert t.rt == sum(t.real_presentation_times)
end
