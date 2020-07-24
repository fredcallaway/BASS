using Optim
using StatsFuns: normcdf, normpdf

"Directed Cognition as defined by Gabaix and Laibson (2005).

The DC policy does a limited kind of look-ahead by considering different
amounts of additional sampling that it could commit to. It estimates the VOC
for taking one additional sample to be max_N VOC(take N samples). This is a
lower bound on the true VOC because you don't actually have to commit in
advance.
"
struct DirectedCognition <: Policy
    m::BDDM
end

stop(pol::DirectedCognition, s::State, t::Trial) = voc_dc(pol.m, s, t) < 0


"Directed Cognition approximation to the value of computation."
function voc_dc(m, s, t)
    # note that we treat the number of samples as a continuous variable here
    # and we assume you can't take more than 100
    λ_avg = average_precision(m, t)
    res = optimize(1, 100, abs_tol=m.cost) do n
        -voc_n(m, s, n, λ_avg)
    end
    -res.minimum
end

"Average precision of samples for each item (averaging out attention)."
function average_precision(m::BDDM, t::Trial)
    attention_proportion = mean.(t.presentation_times) 
    attention_proportion ./= sum(attention_proportion)
    base = m.base_precision .* t.confidence 
    @. base * attention_proportion + m.attention_factor * base * (1 - attention_proportion)
end

"Standard deviation of the posterior mean given precisions of the prior and observation."
function std_of_posterior_mean(λ, λ_obs)
    w = λ_obs / (λ + λ_obs)
    σ_sample = √(1/λ + 1/λ_obs)
    w * σ_sample
end

"Expected termination reward in a future belief state with greater precision, λ_future."
function expected_term_reward(μ, λ, risk_aversion, λ_future)
    σ1 = λ[1] ^ -0.5; σ2 = λ[2] ^ -0.5
    # expected subjective values in future belief state
    v1 = μ[1] - risk_aversion * λ_future[1] ^ -0.5; v2 = μ[2] - risk_aversion * λ_future[2] ^ -0.5
    # standard deviation of difference beteween future values
    θ = √(σ1^2 + σ2^2)
    α = (v1 - v2) / θ  # difference scaled by std
    p1 = normcdf(α)  # p(V1 > V2)
    p2 = 1 - p1
    v1 * p1 + v2 * p2 + θ * normpdf(α)
end

"Value of information from n more samples (assuming equal attention)."
function voi_n(m::BDDM, s::State, n::Real, λ_avg::Vector)
    λ_μ = m.tmp
    λ_obs = n .* λ_avg
    for i in eachindex(s.λ)
        λ_μ[i] = std_of_posterior_mean(s.λ[i], λ_obs[i]) ^ -2
    end
    λ_future = s.λ .+ λ_obs
    # σ_μ ≈ 0. && return 0.  # avoid error initializing Normal
    expected_term_reward(s.μ, λ_μ, m.risk_aversion, λ_future) - term_reward(m, s)
end

"Value of computation from n more samples."
voc_n(m::BDDM, s::State, n::Real, λ_avg::Vector) = voi_n(m, s, n, λ_avg) - m.cost * n


# NOT CORRECT
# "Value of perfect information about all items."
# function vpi(s)
#     expected_max_norm(s.μ, s.λ) - maximum(s.μ)
# end

# struct Policy
#     β_risk::Float64
#     β_vpi::Float64
#     intercept::Float64
# end

# function stop(pol::Policy, s::State)
#     choice = argmax(s.μ)
#     risk = s.λ[choice] ^ -0.5
#     pol.β_risk * risk + pol.β_vpi * vpi(s) + pol.intercept < 0
# end