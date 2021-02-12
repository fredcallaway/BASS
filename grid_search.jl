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
# %% --------

function grid_search(model, version, box, grid_size;
        ε=.01, tol=0, repeats=10, min_multiplier=1.2)
    path = "tmp/$(lowercase(string(model)))/grid/$version"
    println("Writing results to $path")
    mkpath(path)
    
    candidates = map(grid(grid_size, box)) do g
        model(;g...)
    end

    map(pairs(group(d->d.subject, all_data))) do (subj, data)
        out = "$path/$subj"
        if isfile(out)
            println("$out already exists")
            return
        end

        println("Fitting subject $subj")
        trials = prepare_trials(Table(data); dt=.025)
        filter!(trials) do t
            # this can happen due to rounding error
            t.rt <= max_rt(t)
        end

        ibs_kws = (;ε, tol, repeats, min_multiplier)
        results = @showprogress pmap(candidates) do m
            ibs_loglike(m, trials[1:2:end]; ibs_kws...)
        end
        chance = chance_loglike(trials[1:2:end]; ibs_kws.tol)
        serialize(out, (;box, trials, results, chance, ibs_kws))
        # return results
    end
end

# %% --------

box = Box(
    base_precision = (.005, .2, :log),
    attention_factor = (0, 2),
    cost = (.001, .003),
    risk_aversion = (0, .2),
)

grid_search(BDDM, "v1", box, 10; repeats=50, ε=.05)



# %% ==================== OLD ====================

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


# chance_loglike(trials[1:100]; tol=2)