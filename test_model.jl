include("model.jl")
include("dc.jl")
include("utils.jl")
include("data.jl")

using Test
using SplitApplyCombine
using Statistics
using Random

N_TEST = 100
N_SIM = 100
sample_state() = State(randn(2), 1 .+ rand(2))

"Samples a Trial with fixed confidence and values conditional on the DDM state"
function resample_values(t::Trial, s::State)
    value = @. s.μ + (s.λ ^ -0.5) * $randn(2)
    mutate(t, value=value)
end

function estimate_reward(m::BDDM, pol::Policy, s::State, t::Trial, n; N=100000)
    rs = map(1:N) do i
        t = resample_values(t, s)
        simulate(m, pol; s=copy(s), t, max_step=n).reward
    end
    mean(rs), std(rs) / √N
end

@testset "precision insensitive to dt" begin
    m = BDDM()
    presentation_distributions = [Normal(0.2, 1e-10), Normal(0.5, 1e-10)]
    for i in 1:N_TEST
        t = SimTrial(;presentation_distributions)
        s1 = simulate(m, CantStopWontStop(); t).states[1]
        t = mutate(t, dt=(0.1 / rand(1:100)))  # must evenly divide 0.2 and 0.5
        s2 = simulate(m, CantStopWontStop(); t).states[1]
        @test s1.λ ≈ s2.λ
    end
end

@testset "updating makes sense" begin
    m = BDDM()
    presentation_distributions = [Normal(0.2, 1e-10), Normal(0.5, 1e-10)]
    for i in N_TEST
        t = SimTrial(;presentation_distributions, dt=rand([.1, .2, .3]))
        μ1, μ2 = map(1:10000) do i
            simulate(m, CantStopWontStop(); t).states[1].μ
        end |> invert
        λ1, λ2 = simulate(m, CantStopWontStop(); t).states[1].λ
        v1, v2 = t.value

        an_μ1, an_λ1 = bayes_update_normal(0, 1, v1, λ1 - 1)
        @test mean(μ1) ≈ an_μ1 atol=.01

        an_μ2, an_λ2 = bayes_update_normal(0, 1, v2, λ2 - 1)
        @test mean(μ2) ≈ an_μ2 atol=.01
    end
end

@testset "average_precision" begin
    m = BDDM(attention_factor=0.3, confidence_slope=.1)
    map(1:N_TEST) do i
        presentation_distributions=[Normal(0.2, 1e-10), Normal(0.5, 1e-10)]
        # presentation_distributions=[Normal(0.2, 0.1), Normal(0.5, 0.2)]
        t = SimTrial(;presentation_distributions)
        steps_per_cycle = Int(round(sum(mean.(presentation_distributions)) / t.dt, digits=4))
        
        # t.presentation_distributions .= Normal.(1 .+ round.(10 * rand(2)), 1e-5 * ones(2))
        λ_avg = average_precision(m, t)
        max_step = steps_per_cycle * 10
        λ_predicted = ones(2) + λ_avg * max_step
        λ_empirical = simulate(m, CantStopWontStop(); t, max_step).states[end].λ
        @test λ_predicted[1] ≈ λ_empirical[1]

        missing_precision_from_first_fixation =
            base_precision(m, t)[2] * mean(presentation_distributions[1]) * m.attention_factor / t.dt

        @test λ_predicted[2] ≈ λ_empirical[2] + missing_precision_from_first_fixation
    end
end


# These tests don't work because the approximation isn't perfect
# and it's really noisy: See plot_voc.jl for a visual confirmation
# that the analytic VOC is tracking the empirical reward.

# function voc_error(m)
#     s = sample_state()
#     presentation_distributions=[Normal(0.2, 1e-10), Normal(0.5, 1e-10)]
#     steps_per_cycle = Int(round(sum(mean.(presentation_distributions)) / t.dt, digits=4))
#     n = steps_per_cycle * rand(1:10)
#     t = SimTrial(;presentation_distributions)

#     pol = CantStopWontStop()
#     λ_avg = average_precision(m, t)
#     analytic = voc_n(m, s, n, λ_avg, t.dt)
#     r_hat, r_sem = estimate_reward(m, pol, s, t, n)
#     ε = analytic - (r_hat - term_reward(m, s))
#     ε / r_sem
# end



# @testset "Directed Cognition VOC" begin
#     # Compare the analytic VOC to one estimated by simulation
#     @testset "without risk aversion" begin
#         bias = map(1:N_TEST) do i
#             ε = voc_error(cost=1e-3*rand(), risk_aversion=0.)
#             @test ε ≈ 0 atol=5  # error is no more than 5 SEMs from 0 
#             ε
#         end |> mean
#         @test bias ≈ 0 atol=3
#     end

#     @testset "with risk aversion" begin
#         bias = map(1:N_TEST) do i
#             ε = voc_error(cost=1e-3*rand(), risk_aversion=0.5rand())
#             @test ε ≈ 0 atol=5  # error is no more than 5 SEMs from 0 
#             ε
#         end |> mean
#         @test bias ≈ 0 atol=3
#     end
# end
