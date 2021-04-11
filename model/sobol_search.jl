function sobol_search(box, N, data, out; dt=.025,
        ε=.05, tol=0, repeats=10, min_multiplier=1.2)    

    if isfile(out)
        println("$out already exists")
        return
    end

    xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect
    trials = prepare_trials(Table(data); dt)
    filter!(trials) do t
        # this can happen due to rounding error
        t.rt <= max_rt(t)
    end

    ibs_kws = (;ε, tol, repeats, min_multiplier)
    results = @showprogress "$out  " pmap(xs) do x
        m = BDDM(;box(x)...)
        ibs_loglike(m, trials[1:2:end]; ibs_kws...)
    end
    chance = chance_loglike(trials[1:2:end]; ibs_kws.tol)
    default = delete(ntfromstruct(BDDM()), :tmp, :N)
    serialize(out, (;box, xs, trials, results, chance, ibs_kws, default, dt))
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
