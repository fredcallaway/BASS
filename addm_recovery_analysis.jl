include("figure.jl")

# %% --------

function plot_recovery_marginals(run_name)
    tmp_path = "tmp/$run_name/"
    for subject in readdir(tmp_path)
        plot_recovery_marginals(run_name, subject)
    end
end

function plot_recovery_marginals(run_name, subject)
    
    tmp_path = "tmp/$run_name/"
    figs_path = "figs/$run_name/"
    @unpack box, results, chance = deserialize("tmp/$run_name/$subject")

    candidates = map(grid(10, box)) do g
        ADDM(;g...)
    end
    true_model = candidates[parse(Int, subject)]

    loss = -getfield.(results, :logp)
    chance = -chance

    marginals = map(dimnames(loss)) do d
        dims = setdiff(dimnames(loss), [d])
        minimum(loss; dims) |> dropdims(dims...)
    end
    figure("$run_name/$subject") do
        ps = map(dimnames(loss), axiskeys(loss), marginals) do name, x, l
            d = box[name]
            maybelog = :log in d ? (:log,) : ()
            # maybelog = ()
            plot(x, l, xaxis=(string(name), maybelog...))
            hline!([chance], color=:gray, ls=:dash)
            vline!([getfield(true_model, name)], color=:red)
        end
        plot(ps..., size=(600, 600))
    end
    
end
plot_recovery_marginals("addm/recovery2")

# %% ==================== Sanity check default parameters ====================

data = first(group(d->d.subject, all_data))
trials = prepare_trials(Table(data); dt=.025, normalize_value=false)

m = ADDM()
sim_trials = map(trials) do t
    while true
        sim = simulate(m, SimTrial(t); dt=.025, save_fixations=true, max_t=max_rt(t))
        if sim.choice != -1
            return mutate(t; sim.choice, sim.rt, real_presentation_times=sim.fix_times)
        end
    end
end

T = table(sim_trials)

choose_first = map(group(t-> t.order, T)) do trials
    2 - mean(trials.choice)
end

choose_first = map(group(t-> t.order, all_data)) do trials
    2 - mean(trials.choice)
end

