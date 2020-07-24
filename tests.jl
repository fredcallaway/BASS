include("model.jl")
include("dc.jl")
include("utils.jl")

using Test
using SplitApplyCombine
using Statistics

N_TEST = 100
N_SIM = 100
sample_state() = State(randn(2), 1 .+ rand(2))

"Samples a Trial with fixed confidence and values conditional on the DDM state"
function resample_values(t::Trial, s::State)
    value = @. s.μ + (s.λ ^ -0.5) * $randn(2)
    mutate(t, value=value)
end

function estimate_reward(m::BDDM, pol::Policy, s::State, t::Trial, max_rt; N=100000)
    rs = map(1:N) do i
        t = resample_values(t, s)
        simulate(m, pol; s=copy(s), t=t, max_rt=max_rt).reward
    end
    mean(rs), std(rs) / √N
end

@testset "average_precision" begin
    m = BDDM()
    map(1:N_TEST) do i
        t = Trial()
        t.presentation_times .= Normal.(1 .+ round.(10 * rand(2)), 1e-5 * ones(2))
        λ_avg = average_precision(m, t)
        max_rt = 1000
        λ_predicted = ones(2) + λ_avg * max_rt
        λ_empirical = simulate(m, CantStopWontStop(); t=t, max_rt=max_rt).states[end].λ
        ε = λ_predicted .- λ_empirical
        percent_error = sum(abs.(ε ./ λ_empirical))
        @test percent_error < 0.1
    end
end

function voc_error(;n=rand(100:300), s=sample_state(), t=trial(), kws...)
    m = BDDM(;kws...)
    pol = CantStopWontStop()
    λ_avg = average_precision(m, t)
    analytic = voc_n(m, s, n, λ_avg)
    r_hat, r_sem = estimate_reward(m, pol, s, t, n)
    ε = analytic - (r_hat - term_reward(m, s))
    ε / r_sem
end

@testset "Directed Cognition VOC" begin
    # Compare the analytic VOC to one estimated by simulation
    @testset "without risk aversion" begin
        bias = map(1:N_TEST) do i
            ε = voc_error(cost=1e-3*rand())
            @test ε ≈ 0 atol=5  # error is no more than 5 SEMs from 0 
            ε
        end |> mean
        @test bias ≈ 0 atol=3
    end

    @testset "with risk aversion" begin
        bias = map(1:N_TEST) do i
            ε = voc_error(cost=1e-3*rand(), risk_aversion=rand())
            @test ε ≈ 0 atol=5  # error is no more than 5 SEMs from 0 
            ε
        end |> mean
        @test bias ≈ 0 atol=3
    end
end
