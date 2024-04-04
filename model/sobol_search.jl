function sobol_search(box, N, data, out; dt=.025, ε=.05, tol=0, repeats=10, min_multiplier=1.2)

    if isfile(out)
        println("$out already exists")
        return
    end

    xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect
    trials = prepare_trials(Table(data); dt)
    ibs_kws = (;ε, tol, repeats, min_multiplier)
    models = map(xs) do x
        BDDM(;box(x)...)
    end
    results = @showprogress "$out  " pmap(models) do m
        ibs_loglike(m, trials; ibs_kws...)
    end
    chance = chance_loglike(trials; ibs_kws.tol)
    serialize(out, (;box, xs, models, trials, results, chance, ibs_kws, dt))
end

function run_sobol_ind(model, version, box, N; data=load_human_data(), kws...)
    path = "tmp/$(lowercase(string(model)))/sobol/$version"
    mkpath(path)

    @assert model == BDDM
    
    map(pairs(group(d->d.subject, data))) do (subj, subj_data)
        sobol_search(box, N, subj_data, "$path/$subj"; kws...)
    end
end

function run_sobol_group(model, version, box, N; data=load_human_data(), kws...)
    path = "tmp/$(lowercase(string(model)))/sobol/$version"
    mkpath(path)

    @assert model == BDDM
    
    sobol_search(box, N, data, "$path/group"; kws...)
end
