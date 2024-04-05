import Turing: Turing, @model, @addlogprob!, sample, NUTS, Chains, arraydist, generated_quantities
using Optim
using Dictionaries
using ProgressMeter
using StaticArrays
using DataFrames

# group = SplitApplyCombine.group

half_normal(σ) = Truncated(Normal(0, σ), 0, Inf)

struct Prior{M}
    x::M
end

function Base.display(p::Prior{M}) where M
    println("Prior{$(M.name.name)}")
    fns = fieldnames(M)
    npad = maximum(length ∘ string, fns) + 2
    foreach(fieldnames(M), fieldvalues(p.x)) do k, v
        println("  ", rpad(k, npad), v)
    end
end

function Base.show(io::IO, p::Prior{M}) where M
    print(io, "Prior{$(M.name.name)}()")
end

function Prior(M::Type; kws...)
    ps = map(fieldnames(M)) do variable
        get(kws, variable, prior(M, Val(variable)))
    end
    Prior(M(ps...))
end

mutate(p::Prior; kws...) = Prior(mutate(p.x; kws...))

function paramdist(p::Prior)
    filter(v->v isa Distribution, fieldvalues(p.x)) |> SVector |> arraydist
end

function Random.rand(rng::AbstractRNG, s::Random.SamplerTrivial{<:Prior})
    p = s[]
    instantiate(p, rand(paramdist(p)))
end

function prior(variable::Symbol)
    v = string(variable)
    startswith(v, "ε") ? Beta(1,19) :
    Normal(0, 10) # default
end
prior(M::Type, variable::Symbol) = prior(variable)
prior(M::Type, ::Val{variable}) where {variable} = prior(M, variable)


function turing_model_factory(model_prior::Prior; mode=:full)
    @model turing_model(trials, ::Type{T} = Float64) where T = begin
        θ ~ paramdist(model_prior)
        model = instantiate(model_prior, θ)
        if !isnothing(trials)
            @addlogprob! log_likelihood(model, trials; mode)
        end
        return model
    end
end

function instantiate(p::Prior{M}, vals::AbstractVector) where M
    itervals = Iterators.Stateful(vals)
    full = map(fieldvalues(p.x)) do v
        v isa Distribution ? first(itervals) : v
    end
    base_class = M.name.wrapper
    base_class(full...)
end

struct Posterior
    prior
    mode
    trials
    samples
    lp
end

function Base.display(post::Posterior)
    M = basetype(typeof(post.prior.x))
    println("Posterior{$(M)}")
    fns = fieldnames(M)
    npad = maximum(length ∘ string, fns) + 2

    foreach(fieldnames(M)) do fn
        x = get.(post.samples, fn)
        u = unique(x)
        if length(u) == 1
            println("  ", rpad(fn, npad), "$(only(u))")
        else
            m = mean(x)
            q1, q2 = quantile(x, [.1, .9])
            m, q1, q2 = map([m, q1, q2]) do z
                round(z; sigdigits=3)
            end
            println("  ", rpad(fn, npad), "$m [$q1, $q2]")
        end
    end
end

function Base.show(io::IO, post::Posterior)
    M = basetype(typeof(post.prior.x))
    print(io, "Posterior{$(M)}()")
end

function fit2(p::Prior, trials::Vector; mode=:full, n_sample=1000)
    tm = turing_model_factory(p; mode)
    chn = sample(tm(trials), NUTS(), n_sample)
    samples = generated_quantities(tm(nothing), chn)
    Posterior(p, mode, trials, samples, chn[:lp])
end

function simulate(post::Posterior; repeats=1)
    map(post.samples, repeat(post.trials, repeats)) do model, trial
        simulate(model, trial; post.mode)
    end
end

GroupedTrials = Dictionary{String, <:Vector}
GroupedPosterior = Dictionary{String, Posterior}

function StatsBase.fit(p::Prior, gtrials::GroupedTrials; mode=:full)::GroupedPosterior
    @showprogress pmap(gtrials) do trials
        fit(p, trials; mode)
    end
end

function simulate(mp::GroupedPosterior; repeats=1)
    flatmap(mp) do post
        simulate(post; repeats)
    end
end

function fit_map(p::Prior, trials::Vector; mode=:full, repeats=10, time_limit=2)
    model = turing_model_factory(p; mode)(trials)
    results = repeatedly(repeats) do
        optimize(model, Turing.MAP(), Optim.Options(;time_limit))
    end
    lps = get.(results, :lp)
    if sum(lps .≈ maximum(lps)) < 2
        @warn "unstable optimization" trials[1].pid
    end
    argmax(get(:lp), results)
end


function fit_map(p::Prior, gtrials::GroupedTrials; mode=:full, kws...)
    @showprogress pmap(gtrials) do trials
        fit_map(p, trials; mode, kws...)
    end
end

function DataFrames.DataFrame(post::Posterior)
    df = DataFrame(named_tuple.(post.samples)[:])
    insertcols!(df, 1, :lp => post.lp[:])
end

function DataFrames.DataFrame(mp::GroupedPosterior)
    flatmap(keys(mp), values(mp)) do pid, post
        df = DataFrame(post)
        insertcols!(df, 1, :pid => fill(pid, nrow(df)))
    end
end

# function ppc_frame(func, human_trials, model_trials)
#     vcat(
#         @transform!(func(human_trials), :agent = "human"),
#         @transform!(func(model_trials), :agent = "model"),
#     )
# end


# function DataFrames.DataFrame(...)
#     map_fits = @chain gtrials begin
#          @showprogress pmap(_) do trials
#         end
#         DataFrame
#         @rselect begin
#             $AsTable = namedtuple(Dict(:values))
#             :lp
#             :converged = Optim.converged(:optim_result)
#         end
#         @transform :pid = collect(pids)
#     end
# end