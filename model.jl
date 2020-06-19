using StatsFuns: normcdf, normpdf
using Parameters

# ---------- Basics ---------- #

"Bayesian Drift Diffusion Model"
struct BDDM
    N::Int   # num items
    base_precision::Float64  # precision of sample of attended item with confidence=1
    attention_factor::Float64  # < down-weighting of precision for unattended item (less than 1)
    cost::Float64  # cost per sample
    risk_aversion::Float64  # scales penalty for variance of chosen item
    tmp::Vector{Float64}  # implementation detail, for memory-efficiency
end

function BDDM(;N=2, base_precision=.05, attention_factor=.1, cost=1e-3, risk_aversion=0.)
    BDDM(N, base_precision, attention_factor, cost, risk_aversion, zeros(N))
end

"The state of the BDDM."
struct State
    μ::Vector{Float64}
    λ::Vector{Float64}
end
State(n::Int) = State(zeros(n), ones(n))
State(m::BDDM) = State(m.N)
Base.copy(s::State) = State(copy(s.μ), copy(s.λ))

"A single choice trial"
struct Trial
    value::Vector{Float64}
    confidence::Vector{Float64}
    presentation_times::Vector{Int}
end
Trial(n) = Trial(randn(n), 0.1 * ones(n) + 0.9 * rand(n), [1, 1])
Trial(m::BDDM) = Trial(m.N)

include("policy.jl")

# ---------- Updating ---------- #

"Returns updated mean and precision given a prior and observation."
function bayes_update_normal(μ, λ, obs, λ_obs)
    λ1 = λ + λ_obs
    μ1 = (obs * λ_obs + μ * λ) / λ1
    (μ1, λ1)
end

"Precision for each item given the rating confidence and attention."
function observation_precision(m::BDDM, confidence::Vector{Float64}, attended_item::Int)
    λ = m.tmp  # use pre-allocated array for efficiency
    for i in eachindex(λ)
        weight = i == attended_item ? 1. : m.attention_factor
        λ[i] = m.base_precision * confidence[i] * weight
    end
    λ
end

"Take one step of the BDDM.

Draws samples of each item, centered on their true values with precision based
on confidence and attention. Integrates these samples into the current belief
State by Bayesian inference.
"
function update!(m::BDDM, s::State, t::Trial, attended_item::Int)
    λ_obs = observation_precision(m, t.confidence, attended_item)
    for i in eachindex(t.value)
        σ_obs = λ_obs[i] ^ -0.5
        obs = t.value[i] + σ_obs * randn()
        s.μ[i], s.λ[i] = bayes_update_normal(s.μ[i], s.λ[i], obs, λ_obs[i])
    end
end

"Reward attained when terminating sampling.

Each item's value is penalized by its uncertainty (a non-standard form of risk
aversion). The item with maximal (risk-discounted) value is chosen, and the
(risk-discounted) expected value of the item is received as a reward.
"
function term_reward(m::BDDM, s::State)
    maximum(subjective_values(m, s))
end

function subjective_values(m::BDDM, s::State)
    v = m.tmp  # use pre-allocated array for efficiency
    @. v = s.μ - m.risk_aversion * s.λ ^ -0.5
end

# ---------- Simulation ---------- #
"Simulates a choice trial with a given BDDM and stopping Policy."
function simulate(m::BDDM, pol::Policy; t=Trial(m), s=State(m), max_rt=1000)
    items = Iterators.Stateful(Iterators.cycle(1:m.N))
    ptimes = Iterators.Stateful(Iterators.cycle(t.presentation_times))
    attended_item = first(items)
    time_to_switch = first(ptimes)
    rt = 0
    while rt < max_rt
        rt += 1
        if time_to_switch == 0
            attended_item = popfirst!(items)
            time_to_switch = popfirst!(ptimes)
        end
        update!(m, s, t, attended_item)
        time_to_switch -= 1
        stop(pol, s, t) && break
    end
    value, choice = findmax(subjective_values(m, s))
    reward = value - rt * m.cost
    (choice=choice, rt=rt, reward=reward, final_state=s)
end
