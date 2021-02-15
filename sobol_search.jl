@time @everywhere begin
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

# %% --------

function sobol_search(model, version, box, N; data=load_human_data(), dt=.025,
        ε=.05, tol=0, repeats=10, min_multiplier=1.2)
    path = "tmp/$(lowercase(string(model)))/sobol/$version"
    println("Writing results to $path")
    mkpath(path)
    
    xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect

    map(pairs(group(d->d.subject, data))) do (subj, subj_data)
        out = "$path/$subj"
        if isfile(out)
            println("$out already exists")
            return
        end

        trials = prepare_trials(Table(subj_data); dt)
        filter!(trials) do t
            # this can happen due to rounding error
            t.rt <= max_rt(t)
        end

        ibs_kws = (;ε, tol, repeats, min_multiplier)
        results = @showprogress out pmap(xs) do x
            m = BDDM(;box(x)...)
            ibs_loglike(m, trials[1:2:end]; ibs_kws...)
        end
        chance = chance_loglike(trials[1:2:end]; ibs_kws.tol)
        default = delete(ntfromstruct(BDDM()), :tmp, :N)
        serialize(out, (;box, xs, trials, results, chance, ibs_kws, default, dt))
    end
end

# %% --------
box = Box(
    base_precision = (.01, 2, :log),
    attention_factor = (0, 2),
    cost = (.01, .1, :log),
    confidence_slope = (0, 1),
    prior_mean = (-1, 1),
)

sobol_search(BDDM, "test", box, 5000, repeats=10, dt=.1, tol=4)




# %% ==================== Hacking ====================



# %% --------
data = group(d->d.subject, all_data) |> first
trials = prepare_trials(Table(data); dt=.025)
m = BDDM()
@time ibs_loglike(m, trials[1:2:end]; ε=.1, tol=0, repeats=10, min_multiplier=1.2)

# %% --------
xs = Iterators.take(SobolSeq(n_free(box)), 1000) |> collect
results = @showprogress pmap(xs) do x
    m = BDDM(;box(x)...)
    ibs_loglike(m, trials[1:2:end]; ibs_kws....)
end
subj = data[1].subject
chance = chance_loglike(trials[1:2:end]; tol=0)
mkpath("tmp/bddm/sobol/v5")
serialize("tmp/bddm/sobol/v5/$subj", (;box, trials, results, chance, ibs_kws))

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
