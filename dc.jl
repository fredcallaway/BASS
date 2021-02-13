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
    λ_avg::Vector{Float64}
    noise::Logistic
end

# DirectedCognition(m::BDDM) = DirectedCognition(m, zeros(m.N), DiracDelta(0.))
DirectedCognition(m::BDDM, β=1e10) = DirectedCognition(m, zeros(m.N), Logistic(0., 1/β))

function initialize!(pol::DirectedCognition, t)
    pol.λ_avg .= average_precision(pol.m, t)
end

stop(pol::DirectedCognition, s::State, t::Trial) = !voc_is_positive(pol, s, t, rand(pol.noise))
# stop(pol::DirectedCognition, s::State, t::Trial) = voc_dc(pol.m, s, t) < 0

"Directed Cognition approximation to the value of computation."
function voc_dc(m, s, t)
    # note that we treat the number of samples as a continuous variable here
    # and we assume you can't take more than 100
    λ_avg = average_precision(m, t)
    res = optimize(1, 100, GoldenSection(), abs_tol=1) do n  # note abs_tol is on the number of samples
        -voc_n(m, s, n, λ_avg, t.dt)
    end
    -res.minimum
end

"Short-circuit voc"
function voc_is_positive(pol::Policy, s, t, offset)
    @unpack λ_avg, m = pol
    voc_n(m, s, 1, λ_avg, t.dt) > 0 && return true
    # NOTE: the target keyword relies on my fork of Optim.jl
    # https://github.com/fredcallaway/Optim.jl/
    res = optimize(1., 100., abs_tol=1., target=offset) do n
        -voc_n(m, s, n, λ_avg, t.dt)
    end
    # Slower short circuit, works with stable Optim.jl
    # res = optimize(1, 100, abs_tol=1, callback = x-> x.value < offset) do n
    #     -voc_n(m, s, n, λ_avg)
    # end
    return res.minimum < offset
end


"Average precision of samples for each item (averaging out attention)."
function average_precision(m::BDDM, t::Trial)
    attention_proportion = mean.(t.presentation_times)
    attention_proportion ./= sum(attention_proportion)
    base = base_precision(m, t)
    @. base * attention_proportion + m.attention_factor * base * (1 - attention_proportion)
end

"Standard deviation of the posterior mean given precisions of the prior and observation."
function std_of_posterior_mean(λ, λ_obs)
    w = λ_obs / (λ + λ_obs)
    σ_sample = √(1/λ + 1/λ_obs)
    w * σ_sample
end

"Expected termination reward in a future belief state with greater precision, λ_future."
function expected_term_reward(μ1, μ2, σ1, σ2, λ_future1, λ_future2, risk_aversion)
    # expected subjective values in future belief state
    v1 = μ1 - risk_aversion * λ_future1 ^ -0.5
    v2 = μ2 - risk_aversion * λ_future2 ^ -0.5
    # standard deviation of difference beteween future values
    θ = √(σ1^2 + σ2^2)
    α = (v1 - v2) / θ  # difference scaled by std
    p1 = normcdf(α)  # p(V1 > V2)
    p2 = 1 - p1
    v1 * p1 + v2 * p2 + θ * normpdf(α)
end

# function expected_term_reward(μ, λ, risk_aversion, λ_future)
#     σ1 = λ[1] ^ -0.5; σ2 = λ[2] ^ -0.5
#     # expected subjective values in future belief state
#     v1 = μ[1] - risk_aversion * λ_future[1] ^ -0.5; v2 = μ[2] - risk_aversion * λ_future[2] ^ -0.5
#     # standard deviation of difference beteween future values
#     θ = √(σ1^2 + σ2^2)
#     α = (v1 - v2) / θ  # difference scaled by std
#     p1 = normcdf(α)  # p(V1 > V2)
#     p2 = 1 - p1
#     v1 * p1 + v2 * p2 + θ * normpdf(α)
# end

"Value of information from n more samples (assuming equal attention)."
function voi_n(m::BDDM, s::State, n::Real, λ_avg::Vector)
    σ1 = std_of_posterior_mean(s.λ[1], n * λ_avg[1])
    σ2 = std_of_posterior_mean(s.λ[2], n * λ_avg[2])

    λ_future1 = s.λ[1] + n * λ_avg[1]
    λ_future2 = s.λ[2] + n * λ_avg[2]
    # σ_μ ≈ 0. && return 0.  # avoid error initializing Normal
    expected_term_reward(s.μ[1], s.μ[2], σ1, σ2, λ_future1, λ_future2, m.risk_aversion) - term_reward(m, s)
end

"Value of computation from n more samples."
voc_n(m::BDDM, s::State, n::Real, λ_avg::Vector, dt::Float64) = voi_n(m, s, n, λ_avg) - dt * m.cost * n


struct MetaGreedy <: Policy
    m::BDDM
    λ_avg::Vector{Float64}
    n::Int
end
MetaGreedy(m, n=1) = MetaGreedy(m, zeros(m.N), n)

function initialize!(pol::MetaGreedy, t)
    pol.λ_avg .= average_precision(pol.m, t)
end

function stop(pol::MetaGreedy, s::State, t::Trial)
    voc_n(pol.m, s, pol.n, pol.λ_avg, t.dt) <= 0
end

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