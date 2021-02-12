@everywhere begin
    include("utils.jl")
    include("model.jl")
    include("dc.jl")
    include("data.jl")
    include("likelihood.jl")
    include("ibs.jl")
    include("box.jl")
    using Serialization
end
using ProgressMeter
using Sobol
using SplitApplyCombine
@everywhere 
# %% --------


box = Box(
    base_precision = (.005, .2, :log),
    attention_factor = (0, 2),
    cost = (.001, .003),
    risk_aversion = (0, .2),
)

data = group(d->d.subject, all_data) |> first
trials = prepare_trials(Table(data); dt=.025)
# m = BDDM()
# @time ibs_loglike(m, trials[1:2:end]; ε=.01, tol=0, repeats=1, min_multiplier=1)

# %% --------
xs = Iterators.take(SobolSeq(n_free(box)), 10000) |> collect
@everywhere trials = $trials
@everywhere box = $box
results = @showprogress pmap(xs) do x
    m = BDDM(;box(x)...)
    ibs_loglike(m, trials[1:2:end]; ε=.05, tol=0, repeats=50, min_multiplier=1.2)
end
subj = data[1].subject
chance = chance_loglike(trials[1:2:end]; tol=0)
mkpath("tmp/bddm/sobol/v3")
serialize("tmp/bddm/sobol/v3/$subj", (;box, xs, results, chance))


# %% --------
X = invert(xs[partialsortperm(nll, 1:100)])
q = map(X) do x
    quantile(x, [.05, .2, .8, .95])
end |> invert
convert(Dict, invert(map(box, q)))


# %% --------

nll = [-r.logp for r in results]
top = partialsortperm(nll, 1:100)
nll[top]

fx2 = @showprogress pmap(xs) do x
    m = BDDM(;box(x)...)
    ibs_loglike(m, trials[1:2:end]; ε=.01, tol=0, repeats=50, min_multiplier=1)
end



# %% --------

# @everywhere out = "tmp/ibs_sobol_$(trials[1].subject)"
results = @showprogress pmap(enumerate(candidates)) do (i, m)
    result = ibs_loglike(m, trials[1:2:end]; ε=.5, tol=10, repeats=100, min_multiplier=2)
    # serialize("$out/$i", (;m, result))
    result
end
chance = chance_loglike(trials[1:2:end]; tol=10)
serialize("tmp/sobol_1", (;box, xs, results, chance))

# %% --------

logp, converged = invert(results)
best = partialsortperm(logp, 1:10)

map(candidates[best]) do m
    (;m.base_precision, m.attention_factor, m.cost, m.risk_aversion)
end |> Table

map(candidates[best]) do m
    ibs_loglike(m, trials[1:2:end]; ε=.5, tol=10, repeats=10, min_multiplier=2).logp
end
