include("model.jl")
using Test
using SplitApplyCombine
using Statistics

N_TEST = 100

sample_state() = State(randn(2), 1 .+ rand(2))

"Samples a Trial with fixed confidence and values conditional on the DDM state"
function sample_trial(s::State, confidence)
    value = @. s.μ + (s.λ ^ -0.5) * $randn(2)
    presentation_times = ones(2)  # NOTE: the directed cognition VOC assumes this
    Trial(value, confidence, presentation_times)
end

function estimate_reward(m::BDDM, pol::Policy, s::State, confidence, max_rt; N=100000)
    rs = map(1:N) do i
        t = sample_trial(s, confidence)
        simulate(m, pol; s=copy(s), t=t, max_rt=max_rt).reward
    end
    mean(rs), std(rs) / √N
end

@testset "DC VOC without risk aversion" begin
    # Compare the analytic VOC to one estimated by simulation
    pol = CantStopWontStop()
    εs = map(1:N_TEST) do i
        m = BDDM(cost=1e-3*rand())
        s = sample_state()
        conf = Trial(m).confidence
        n = rand(10:30)
        analytic = voc_n(m, s, n, conf)
        r_hat, r_sem = estimate_reward(m, pol, s, conf, n)
        ε = analytic - (r_hat - term_reward(m, s))
        @test abs(ε / r_sem) < 5  # error is no more than 5 SEMs from 0 
        ε / r_sem
    end
    println("Bias:", round(mean(εs); digits=2))
    # @test mean(εs) 
    # @test mean(as) ≈ mean(es) atol = 1e-3  # check for systematic bias
end

@testset "DC VOC with risk aversion" begin
    pol = CantStopWontStop()
    εs = map(1:N_TEST) do i
        m = BDDM(cost=1e-3*rand(), risk_aversion=rand())
        s = sample_state()
        conf = Trial(m).confidence
        n = rand(10:30)
        analytic = voc_n(m, s, n, conf)
        r_hat, r_sem = estimate_reward(m, pol, s, conf, n)
        ε = analytic - (r_hat - term_reward(m, s))
        @test abs(ε / r_sem) < 5  # error is no more than 5 SEMs from 0 
        ε / r_sem
    end
    println("Bias:", round(mean(εs); digits=2))
end
