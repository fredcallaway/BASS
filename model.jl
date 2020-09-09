using Distributions
using Random
using Parameters
# ---------- Basics ---------- #

"Bayesian Drift Diffusion Model"
@with_kw struct BDDM
    N::Int = 2                      # number of items
    base_precision::Float64 = .05   # precision of sample of attended item with confidence=1
    attention_factor::Float64 = .1  # down-weighting of precision for unattended item (less than 1)
    cost::Float64 = 1e-3            # cost per sample
    risk_aversion::Float64 = 0.     # scales penalty for variance of chosen item
    over_confidence_slope::Float64 = 1.   # treat observations as though they were more or less noisy 
    over_confidence_intercept::Float64 = 0.   # treat observations as though they were more or less noisy 
    prior_mean::Float64 = 0.
    prior_precision::Float64 = 1.
    tmp::Vector{Float64} = zeros(N) # implementation detail, for memory-efficiency
end


"The state of the BDDM."
struct State
    μ::Vector{Float64}
    λ::Vector{Float64}
end
State(m::BDDM) = State(fill(m.prior_mean, m.N), fill(m.prior_precision, m.N))
Base.copy(s::State) = State(copy(s.μ), copy(s.λ))

"A single choice trial"
struct Trial
    value::Vector{Float64}
    confidence::Vector{Float64}
    presentation_times::Vector{Distribution}
end

# ---------- Updating ---------- #

"Returns updated mean and precision given a prior and observation."
function bayes_update_normal(μ, λ, obs, λ_obs)
    λ1 = λ + λ_obs
    μ1 = (obs * λ_obs + μ * λ) / λ1
    (μ1, λ1)
end

"Take one step of the BDDM.

Draws samples of each item, centered on their true values with precision based
on confidence and attention. Integrates these samples into the current belief
State by Bayesian inference.
"
function update!(m::BDDM, s::State, true_value::Vector, λ_obs::Vector)
    for i in eachindex(λ_obs)
        λ_obs[i] == 0 && continue  # no update
        σ_obs = λ_obs[i] ^ -0.5
        obs = true_value[i] + σ_obs * randn()
        λ_obs[i] = λ_obs[i] * m.over_confidence_slope + m.over_confidence_intercept
        s.μ[i], s.λ[i] = bayes_update_normal(s.μ[i], s.λ[i], obs, λ_obs[i])
    end
end

# ---------- Choice and value ---------- #

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

# ---------- Attention ---------- #

"Precision for each item given the rating confidence and attention."
function observation_precision(m::BDDM, confidence::Vector, attended_item::Int)
    λ = m.tmp  # use pre-allocated array for efficiency
    for i in eachindex(λ)
        weight = i == attended_item ? 1. : m.attention_factor
        λ[i] = m.base_precision * confidence[i] * weight
    end
    λ
end

function make_switches(presentation_times)
    switching = presentation_times |> enumerate |> Iterators.cycle |> Iterators.Stateful
    function switch()
        i, d = first(switching)
        t = max(1, round(Int, rand(d)))
        i, t
    end
end

# ---------- Stopping policy ---------- #

"A Policy decides when to stop sampling."
abstract type Policy end

"A Policy endorsed by Young Gunz."
struct CantStopWontStop <: Policy end
stop(pol::CantStopWontStop, s::State, t::Trial) = false

# ---------- Simulation ---------- #

"Simulates a choice trial with a given BDDM and stopping Policy."
function simulate(m::BDDM, pol::Policy; t=Trial(), s=State(m), max_rt=1000, save_states=false)
    switch = make_switches(t.presentation_times)
    attended_item, time_to_switch = switch()
    first_fix = true
    rt = 0
    states = []
    presentation_times = zeros(Int, length(t.value))
    while rt < max_rt
        save_states && push!(states, copy(s))
        rt += 1
        if time_to_switch == 0
            attended_item, time_to_switch = switch()
            first_fix = false
        end
        λ_obs = observation_precision(m, t.confidence, attended_item)
        if first_fix
            for i in eachindex(λ_obs)
                if i != attended_item
                    λ_obs[i] = 0
                end
            end
        end
        update!(m, s, t.value, λ_obs)
        presentation_times[attended_item] += 1
        time_to_switch -= 1
        stop(pol, s, t) && break
    end
    value, choice = findmax(subjective_values(m, s))
    reward = value - rt * m.cost
    push!(states, s)
    (choice=choice, rt=rt, reward=reward, states=states, presentation_times=presentation_times)
end


# ---------- Miscellaneous ---------- #

namedtuple(t::Trial) = (
    val1 = t.value[1],
    val2 = t.value[2],
    conf1 = t.confidence[1],
    conf2 = t.confidence[2],
)

namedtuple(m::BDDM) = (
    base_precision = m.base_precision,
    attention_factor = m.attention_factor,
    cost = m.cost,
    risk_aversion = m.risk_aversion,
    over_confidence_slope = m.over_confidence_slope,
    over_confidence_intercept = m.over_confidence_intercept,
    prior_mean = m.prior_mean,
    prior_precision = m.prior_precision
)
