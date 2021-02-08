@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    using Serialization
end
using ProgressMeter
using Sobol
using SplitApplyCombine
include("box.jl")
# # %% --------
# trials = prepare_trials(all_data; dt=.01)
# max_rt = quantile([t.rt for t in trials], .99)
# trials = filter(t-> t.rt ≤ max_rt, trials)

# %% --------

box = Box(
    base_precision = (10^-3, 10^-1, :log),
    attention_factor = (0, 2),
    cost = (10^-4.5, 10^-2.5, :log),
    risk_aversion = (0, .2),
)

# %% ==================== Grid ====================

candidates = map(grid(7, box)) do g
    BDDM(;g...)
end

mkpath("tmp/grid/feb7")
map(pairs(group(d->d.subject, all_data))) do (subj, data)
    println("Fitting subject $subj")
    trials = prepare_trials(Table(data); dt=.025)
    ibs_kws = (ε=.5, tol=10, repeats=100, min_multiplier=1.2)
    results = @showprogress pmap(candidates) do m
        ibs_loglike(m, trials[1:2:end]; ibs_kws...)
    end
    chance = chance_loglike(trials[1:2:end]; tol=10)
    serialize("tmp/grid/feb7/$subj", (;box, trials, results, chance, ibs_kws))
end





# %% ==================== Sobol + GP ====================

xs = Iterators.take(SobolSeq(n_free(box)), 1000)
candidates = map(xs) do x
    BDDM(;box(x)...)
end

# @everywhere out = "tmp/ibs_sobol_$(trials[1].subject)"
results = @showprogress pmap(enumerate(candidates)) do (i, m)
    result = ibs_loglike(m, trials[1:2:end]; ε=.5, tol=10, repeats=100, min_multiplier=2)
    # serialize("$out/$i", (;m, result))
    result
end
chance = chance_loglike(trials[1:2:end]; tol=10)
serialize("tmp/sobol_1", (;box, xs, results, chance))


# %% --------


chance = chance_loglike(trials[1:2:end]; tol=10)
logp, converged = invert(results)
best = partialsortperm(logp, 1:10)

map(candidates[best]) do m
    (;m.base_precision, m.attention_factor, m.cost, m.risk_aversion)
end |> Table

map(candidates[best]) do m
    ibs_loglike(m, trials[1:2:end]; ε=.5, tol=10, repeats=10, min_multiplier=2).logp
end
@time ibs_loglike(m, trials[1:2:end]; ε=.5, tol=10, repeats=10, min_multiplier=2).logp






# %% --------

invert(results).logp .- chance_loglike(trials[1:2:end]; tol=10)



@time ibs_loglike(first(candidates), trials[1:100]; ε=.1, tol=2, repeats=5)



# m = first(candidates)
# ibs(trials[1:100]; repeats=30, min_logp=-Inf) do t
#     is_hit(sample_choice_rt(m, t, 1.), t, 2)
# end
# chance_loglike(trials[1:100]; tol=2)