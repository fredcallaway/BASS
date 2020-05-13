using Optim

abstract type Policy end

struct CantStopWontStop <: Policy end
stop(pol::CantStopWontStop, s::State, t::Trial) = false

struct DirectedCognition <: Policy
    m::BDDM
end

stop(pol::DirectedCognition, s::State, t::Trial) = voc_dc(pol.m, s, t.confidence) < 0

"Directed cognition approximation to the Value of Computation.

It does a limited kind of look-ahead by considering different amounts of
additional sampling that it could commit to. It estimates the VOC for taking
one additional sample to be max_N VOC(take N samples). This is a lower bound
on the true VOC because you don't actually have to commit in advance.
"
function voc_dc(m, s, confidence)
    # note that we treat the number of samples as a continuous variable here
    res = optimize(1, 100, abs_tol=m.cost) do n
        -voc_n(m, s, n, confidence)
    end
    -res.minimum
end

"Standard deviation of the posterior mean"
function std_of_posterior_mean(λ, λ_obs)
    w = λ_obs / (λ + λ_obs)
    σ_sample = √(1/λ + 1/λ_obs)
    w * σ_sample
end

"Expected maximum of Normals with means μ and precisions λ"
function expected_max_norm(μ, λ)
    @assert length(μ) == 2
    Φ = normcdf  # isn't unicode just so much fun!
    ϕ = normpdf
    μ1, μ2 = μ
    σ1 = λ[1] ^ -0.5; σ2 = λ[2] ^ -0.5
    θ = √(σ1^2 + σ2^2)
    p1 = Φ((μ1 - μ2) / θ)  # p(X1 > X2)
    p2 = 1 - p1
    return μ1 * p1 + μ2 * p2 + θ * ϕ((μ1 - μ2) / θ)
end

"Expected maximum plus risk aversion"
function expected_term_reward(μ, λ, risk_aversion, λ_future)
    Φ = normcdf  # isn't unicode just so much fun!
    ϕ = normpdf
    μ1, μ2 = μ
    σ1 = λ[1] ^ -0.5; σ2 = λ[2] ^ -0.5
    θ = √(σ1^2 + σ2^2)
    p1 = Φ((μ1 - μ2) / θ)  # p(X1 > X2)
    p2 = 1 - p1
    emax = μ1 * p1 + μ2 * p2 + θ * ϕ((μ1 - μ2) / θ)
    erisk = p1 * λ_future[1]^-0.5 + p2 * λ_future[2]^-0.5
    emax - risk_aversion * erisk
end


# "Value of information from n more samples (assuming equal attention)"
# function voi_n(m::BDDM, s::State, n, confidence)
#     λ_avg = n * (m.base_precision + m.base_precision * m.attention_factor) / 2
#     λ_μ = m.tmp
#     for i in eachindex(s.λ)
#         λ_μ[i] = std_of_posterior_mean(s.λ[i], λ_avg * confidence[i]) ^ -2
#     end
#     # σ_μ ≈ 0. && return 0.  # avoid error initializing Normal
#     expected_max_norm(s.μ, λ_μ) - maximum(s.μ)
# end


"Value of information from n more samples (assuming equal attention)"
function voi_n(m::BDDM, s::State, n, confidence)
    λ_avg = n * (m.base_precision + m.base_precision * m.attention_factor) / 2
    λ_μ = m.tmp
    for i in eachindex(s.λ)
        λ_μ[i] = std_of_posterior_mean(s.λ[i], λ_avg * confidence[i]) ^ -2
    end
    λ_future = s.λ .+ λ_avg .* confidence
    # σ_μ ≈ 0. && return 0.  # avoid error initializing Normal
    expected_term_reward(s.μ, λ_μ, m.risk_aversion, λ_future) - term_reward(m, s)
end

voc_n(m::BDDM, s::State, n, confidence) = voi_n(m, s, n, confidence) - m.cost * n


"Value of perfect information about all items"
function vpi(s)
    expected_max_norm(s.μ, s.λ) - maximum(s.μ)
end

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