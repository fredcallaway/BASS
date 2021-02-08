include("figure.jl")

# %% --------

function plot_marginals(run_name)
    tmp_path = "tmp/$run_name/"
    for subject in readdir(tmp_path)
        plot_marginals(run_name, subject)
    end
end

function plot_marginals(run_name, subject)
    
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
plot_marginals("addm/recovery")

