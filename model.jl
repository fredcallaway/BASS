using Distributions
using Random
# ---------- Basics ---------- #

"Bayesian Drift Diffusion Model"
struct BDDM
    N::Int   # num items
    base_precision::Float64  # precision of sample of attended item with confidence=1
    attention_factor::Float64  # < down-weighting of precision for unattended item (less than 1)
    cost::Float64  # cost per sample
    risk_aversion::Float64  # scales penalty for variance of chosen item
    over_confidence::Float64  # multiplier for perceived sample variance
    tmp::Vector{Float64}  # implementation detail, for memory-efficiency
end

function BDDM(;N=2, base_precision=.05, attention_factor=.1, cost=1e-3, risk_aversion=0., over_confidence=0)
    BDDM(N, base_precision, attention_factor, cost, risk_aversion, over_confidence, zeros(N))
end

"The state of the BDDM."
struct State
    μ::Vector{Float64}
    λ::Vector{Float64}
end
State(n=2::Int) = State(zeros(n), ones(n))
State(m::BDDM) = State(m.N)
Base.copy(s::State) = State(copy(s.μ), copy(s.λ))

"A single choice trial"
struct Trial
    value::Vector{Float64}
    confidence::Vector{Float64}
    presentation_times::Vector{Distribution}
end

function Trial()
    ptimes = [Normal(10, 2), Normal(30, 6)]
    shuffle!(ptimes)
    Trial(randn(2), 0.1 * ones(2) + 0.9 * rand(2), ptimes)
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
        σ_obs = λ_obs[i] ^ -0.5 * m.over_confidence
        obs = true_value[i] + σ_obs * randn()
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
    over_confidence = m.over_confidence,
)
