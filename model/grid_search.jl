using ProgressMeter

function flexmap(f, args; progress=:default, parallel=false, cache="")
    if progress == :default
        progress = parallel && isa(stderr, Base.TTY)
    end
    mapfun = parallel ? pmap : map
    mkpath(cache)
    func = (i, args) -> begin
        file = "$cache/$i"
        if isfile(file)
            return deserialize(file)
        else
            res = f(args)
            serialize(file, res)
            return res
        end
    end
    if progress
        progress_map(func, eachindex(args), args; mapfun)
    else
        mapfun(func, eachindex(args), args)
    end
end

function compute_likelihoods(out, model, box, grid_size, subj_data; ε, tol, repeats, min_multiplier, parallel=false)
    if isfile(out)
        println("$out already exists")
        return
    end

    # println("Fitting subject $subj")
    trials = prepare_trials(Table(subj_data))
    candidates = map(grid(grid_size, box)) do g
        model(; g...)
    end

    ibs_kws = (; ε, tol, repeats, min_multiplier)
    cache = "tmp/ibs/$out"
    results = flexmap(candidates; parallel, cache) do m
        GC.gc()
        ibs_loglike(m, trials; ibs_kws...)
    end
    chance = chance_loglike(trials; ibs_kws.tol)
    output = (; box, candidates, trials, results, chance, ibs_kws)
    serialize(out, output)
    println("Wrote $out")
    output
end

function grid_search(model, version, box, grid_size, data; group_fit=true,
        ε=0.01, tol=4, repeats=10, min_multiplier=5.0, parallelize=:params
    )
    path = "tmp/$(lowercase(string(model)))/grid/$version"
    println("Writing results to $path")
    mkpath(path)

    if group_fit
        compute_likelihoods("$path/group", model, box, grid_size, data; ε, tol, repeats, min_multiplier, parallel=true)
    else
        flexmap(pairs(group(d -> d.subject, data)), parallel=parallelize == :subjects) do (subj, subj_data)
            GC.gc()  # help out with memory
            compute_likelihoods("$path/$subj", model, box, grid_size, subj_data; ε, tol, repeats, min_multiplier, parallel=parallelize==:params)
        end
    end
end
