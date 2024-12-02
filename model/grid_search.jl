using ProgressMeter

function flexmap(f, args...; progress=:default, parallel=false)
    if progress == :default
        progress = parallel && isa(stderr, Base.TTY)
    end
    mapfun = parallel ? pmap : map
    if progress
        progress_map(f, args...; mapfun)
    else
        mapfun(f, args...)
    end
end

function grid_search(model, version, box, grid_size, data;
        ε=0.01, tol=4, repeats=10, min_multiplier=1.2, parallelize=:params
    )
    path = "tmp/$(lowercase(string(model)))/grid/$version"
    println("Writing results to $path")
    mkpath(path)
    
    flexmap(pairs(group(d -> d.subject, data)), parallel=parallelize == :subjects) do (subj, subj_data)
        GC.gc()  # help out with memory
        out = "$path/$subj"
        if isfile(out)
            println("$out already exists")
            return
        end

        # println("Fitting subject $subj")
        trials = prepare_trials(Table(subj_data); dt=0.025)
        candidates = map(grid(grid_size, box)) do g
            model(; g...)
        end

        ibs_kws = (; ε, tol, repeats, min_multiplier)
        results = flexmap(candidates, parallel=parallelize == :params) do m
            ibs_loglike(m, trials; ibs_kws...)
        end
        chance = chance_loglike(trials; ibs_kws.tol)
        output = (; box, candidates, trials, results, chance, ibs_kws)
        serialize(out, output)
        println("Wrote $out")
        output
    end
end