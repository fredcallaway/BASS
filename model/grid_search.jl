using ProgressMeter

function grid_search(model, version, box, grid_size, data;
    ε=0.01, tol=4, repeats=10, min_multiplier=1.2, progress=isa(stderr, Base.TTY))
    path = "tmp/$(lowercase(string(model)))/grid/$version"
    println("Writing results to $path")
    mkpath(path)

    candidates = map(grid(grid_size, box)) do g
        model(; g...)
    end

    map(pairs(group(d -> d.subject, data))) do (subj, subj_data)
        out = "$path/$subj"
        if isfile(out)
            println("$out already exists")
            return
        end

        println("Fitting subject $subj")
        trials = prepare_trials(Table(subj_data); dt=0.025)

        ibs_kws = (; ε, tol, repeats, min_multiplier)
        results = @showprogress out enabled=progress pmap(candidates) do m
            ibs_loglike(m, trials; ibs_kws...)
        end
        chance = chance_loglike(trials; ibs_kws.tol)
        output = (; box, candidates, trials, results, chance, ibs_kws)
        serialize(out, output)
        println("Wrote $out")
        output
    end
end