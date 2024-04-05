include("base.jl")
include("bayes.jl")


@with_kw struct BDM2{A,B,C,D,E,F,G}
    base_precision::A
    attention_factor::B
    confidence_slope::C = 0.
    subjective_offset::D = 0.
    subjective_slope::E = 1.
    prior_mean::F = 0.
    prior_precision::G = 1.
end

function simulate(m::BDM2, t::Trial; pol::Policy=DirectedCognition(m), s=State(m), max_step=cld(20, t.dt),
                  save_states=false, save_presentation=false)

function total_attention(m, d::NamedTuple)
    pds = d.presentation_duration
    total1 = sum(pds[1:2:end])
    total2 = sum(pds[2:2:end])
    total1_noninit = total1 - pds[1]
    (
        total1 + m.attention_factor * total2,
        total2 + m.attention_factor * total1_noninit
    )
end

"Precision of all information in a trial"
function total_objective_precision(m, d::NamedTuple)
    attention = total_attention(m, d)
    @. attention * (m.base_precision + m.confidence_slope * d.confidence)
end

# %% --------

function posterior_mean_distribution(µ0, λ0, v, λ_obs)
    if λ_obs == 0
        Normal(µ0, 0.)
    else
        @assert λ0 > 0
        @assert λ_obs > 0
        w = λ_obs / (λ0 + λ_obs)
        µ1 = (λ0 * µ0 + v * λ_obs) / (λ0 + λ_obs)
        σ_µ1 = w * λ_obs^-0.5
        Normal(µ1, σ_µ1)
    end
end

function choice_probability(model, d)
    λ_obs = total_objective_precision(model, d)
    µd = @. posterior_mean_distribution(model.prior_mean, model.prior_precision, d.value, λ_obs)
    (µ1, σ1), (µ2, σ2) = params.(µd)
    difference = Normal(µ1 - µ2, √(σ1^2 + σ2^2))
    1 - cdf(difference, 0)
end

function log_likelihood(model::BDM2, trials; mode)
    @assert mode == :choice
    sum(trials) do d
        p = choice_probability(model, d)
        log(d.choice == 1 ? p : 1-p)
    end
end

# %% --------

data = load_human_data(1)
µ, σ = empirical_prior(data)

pp = Prior(BDM2,
    base_precision = Uniform(.01, .1),
    attention_factor = Uniform(0., 2.),
    confidence_slope = 0.,
    subjective_offset = 0.,
    subjective_slope = 1.,
    prior_mean = Uniform(0., µ),
    prior_precision = 1 / σ^2,
    cost = NaN,
)

post = fit2(pp, collect(data); mode=:choice)
df = DataFrame(post)
mean(df.lp)

version = "apr5-choices"


# %% --------

map_bdm = fit_map(pp, collect(data); mode=:choice)
model = instantiate(pp, map_bdm.values)
# write_sim(model, data, version, "1-main")

sim = map(data; repeats=10) do d
    (;d..., choice = rand() < choice_probability(model, d) ? 1 : 2)
end

mkpath("results/apr5-choice")
sim |> CSV.write("results/apr5-choice/1-main.csv")



# %% --------

struct Softmax{A,B}
    β::A
    ε::B
end

function log_likelihood(model::Softmax, trials; mode)
    @assert mode == :choice
    sum(trials) do d
        log(ε * 0.5 + (1 - ε) * softmax(β * d.value)[d.choice])
    end
end

pp_soft = fit2(Prior(Softmax, β=Uniform(0,3), ε=0.), collect(data); mode=:choice)
df_soft = DataFrame(pp_soft)
mean(df_soft.lp)