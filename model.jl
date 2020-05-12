using StatsFuns: normcdf, normpdf

"The state of the BDDM."
struct State
    μ::Vector{Float64}
    λ::Vector{Float64}
end
State(n::Int) = State(zeros(n), ones(n))

"Returns updated mean and precision given a prior and observation."
function bayes_update_normal(μ, λ, obs, λ_obs)
    λ1 = λ + λ_obs
    μ1 = (obs * λ_obs + μ * λ) / λ1
    (μ1, λ1)
end

"Take one step of the BDDM, moving towards the true values"
function update!(s::State, true_values, λ_obs)
    for i in eachindex(true_values)
        σ_obs = λ_obs[i] ^ -0.5
        obs = true_values[i] + σ_obs * randn()
        s.μ[i], s.λ[i] = bayes_update_normal(s.μ[i], s.λ[i], obs, λ_obs[i])
    end
end

function observation_precision(certainty, attended_item; base_λ=.1, attention_factor=0.01)
    attention = ones(length(certainty)) .* (base_λ * attention_factor)
    attention[attended_item] = base_λ
    certainty .* attention
end

"Expected maximum of Normals with means μ and precisions λ"
function expected_max_norm(μ, λ)
    Φ = normcdf  # isn't unicode just so much fun!
    ϕ = normpdf
    if length(μ) == 2
        μ1, μ2 = μ
        σ1, σ2 = λ .^ -0.5
        θ = √(σ1^2 + σ2^2)
        return μ1 * Φ((μ1 - μ2) / θ) + μ2 * Φ((μ2 - μ1) / θ) + θ * ϕ((μ1 - μ2) / θ)
    end

    dists = Normal.(μ, λ.^-0.5)
    mcdf(x) = mapreduce(*, dists) do d
        cdf(d, x)
    end

    - quadgk(mcdf, -10, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, 10, atol=1e-5)[1]
end

"Value of perfect information about all items"
function vpi(s)
    expected_max_norm(s.μ, s.λ) - maximum(s.μ)
end

function stopping(s::State)
    choice = argmax(s.μ)
    risk = s.λ[choice]^-1/2
    return vpi(s) + risk
end

function simulate(true_values, certainty, presentation_times; max_t=1000, threshold=.1)
    N = length(true_values)
    items = Iterators.Stateful(Iterators.cycle(1:N))
    ptimes = Iterators.Stateful(Iterators.cycle(presentation_times))
    attended_item = first(items)
    time_to_switch = first(ptimes)
    s = State(N)
    μs = [copy(s.μ)]
    λs = [copy(s.λ)]
    for t in 1:max_t
        if time_to_switch == 0
            attended_item = popfirst!(items)
            time_to_switch = popfirst!(ptimes)
            # @show attended_item time_to_switch
        end
        λ_obs = observation_precision(certainty, attended_item)
        update!(s, true_values, λ_obs)
        push!(μs, copy(s.μ))
        push!(λs, copy(s.λ))
        time_to_switch -= 1
        stopping(s) < threshold && break
    end
    argmax(s.μ), length(μs), μs, λs
end

function choice_rt(true_values, certainty, presentation_times; n_sim=10000, kws...)
    choice, rt = n_sim \ mapreduce(+, 1:n_sim) do i
        choice, rt, μs, λs = simulate(true_values, certainty, presentation_times; kws...)
        [choice - 1, rt]
    end
    (choice=choice, rt=rt)
end