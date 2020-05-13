using StatsFuns: normcdf, normpdf
using Parameters

# ---------- Basics ---------- #

"Bayesian Drift Diffusion Model"
struct BDDM
    N::Int
    base_precision::Float64
    attention_factor::Float64
    cost::Float64
    risk_aversion::Float64
    tmp::Vector{Float64}  # this is for memory-efficiency
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

"Precision of samples given the rating confidence and attention"
function observation_precision(m::BDDM, confidence, attended_item)
    λ = m.tmp
    for i in eachindex(confidence)
        weight = i == attended_item ? 1. : m.attention_factor
        λ[i] = m.base_precision * confidence[i] * weight
    end
    λ
end

"Take one step of the BDDM, moving towards the true values"
function update!(m::BDDM, s::State, t::Trial, attended_item)
    λ_obs = observation_precision(m, t.confidence, attended_item)
    for i in eachindex(t.value)
        σ_obs = λ_obs[i] ^ -0.5
        obs = t.value[i] + σ_obs * randn()
        s.μ[i], s.λ[i] = bayes_update_normal(s.μ[i], s.λ[i], obs, λ_obs[i])
    end
end

function term_reward(m::BDDM, s::State)
    value, choice = findmax(s.μ)
    value - m.risk_aversion * s.λ[choice] ^ -0.5
end

# ---------- Simulation ---------- #

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
    value, choice = findmax(s.μ)
    σ = s.λ[choice] ^ -0.5
    reward = term_reward(m, s) - rt * m.cost
    (choice=choice, rt=rt, reward=reward, final_state=s)
end
