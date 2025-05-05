using ProgressMeter: progress_map

function flexmap(f, args; progress=:default, parallel=false, cache="")
    if progress == :default
        progress = parallel && isa(stderr, Base.TTY)
    end
    mapfun = parallel ? pmap : map
    mkpath(cache)
    func = (i, args) -> begin
        if cache == ""
            return f(args)
        end
        file = "$cache/$i"
        if isfile(file)
            try
                return deserialize(file)
            catch
                println("Error deserializing $file; removing it")
                rm(file)
            end
        end
        # cache not available
        res = f(args)
        serialize(file, res)
        return res
    end
    if progress
        progress_map(func, eachindex(args), args; mapfun)
    else
        result, t = @timed mapfun(func, eachindex(args), args)
        println("flexmap completed in $t seconds")
        return result
    end
end

function compute_likelihoods(out, model, box, grid_size, subj_data; ε, tol, repeats, min_multiplier, parallel=false)
    if isfile(out)
        println("$out already exists")
        return
    end

    # println("Fitting subject $subj")
    trials = prepare_trials(Table(subj_data))
    # exclude timeouts and very fast trials (< 100ms)
    filter!(t -> 0 < t.rt < MAX_RT / DEFAULT_DT, trials)
    candidates = map(grid(grid_size, box)) do g
        model(; g...)
    end

    ibs_kws = (; ε, tol, repeats, min_multiplier)
    cache = "tmp/ibs/$out"
    chance = chance_loglike(trials; ibs_kws.tol)
    results = flexmap(candidates; parallel, cache) do m
        m.confidence_slope == 0.0 && m.base_precision == 0.0 && return chance
        m.cost == 0.0 && return chance
        GC.gc()
        ibs_loglike(m, trials; ibs_kws...)
    end
    output = (; box, candidates, trials, results, chance, ibs_kws)
    serialize(out, output)
    println("Wrote $out")
    flush(stdout)
    output
end

function grid_search(model, version, box, grid_size, data; fit_mode=:group,
        ε=0.01, tol=0, repeats=10, min_multiplier=2.0, parallelize=:params,
    )
    path = "tmp/$(lowercase(string(model)))/grid/$version"
    println("Writing results to $path")
    mkpath(path)

    if fit_mode == :group
        compute_likelihoods("$path/group", model, box, grid_size, data; ε, tol, repeats, min_multiplier, parallel=true)
    elseif fit_mode == :subject
        flexmap(pairs(group(d -> d.subject, data)), parallel=parallelize == :subjects) do (subj, subj_data)
            GC.gc()  # help out with memory
            compute_likelihoods("$path/$subj", model, box, grid_size, subj_data; ε, tol, repeats, min_multiplier, parallel=parallelize==:params)
        end
    else
        @assert fit_mode isa Int
        subject = fit_mode
        subj_data = collect(group(d -> d.subject, data))[subject]
        compute_likelihoods("$path/$subject", model, box, grid_size, subj_data; ε, tol, repeats, min_multiplier, parallel=true)
    end
end
