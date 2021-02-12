include("figure.jl")
import Missings
# %% --------

function plot_marginals(run_name)
    tmp_path = "tmp/$run_name/"
    figs_path = "figs/$run_name/"
    mkpath(figs_path)

    for subject in readdir(tmp_path)
        @unpack box, results, chance = deserialize(tmp_path * subject)
        loss = -getfield.(results, :logp)
        chance = -chance
        loss = fillmissing.(loss, chance)

        marginals = map(dimnames(loss)) do d
            dims = setdiff(dimnames(loss), [d])
            minimum(loss; dims) |> dropdims(dims...)
        end
        figure("$run_name/$subject") do
            ylim = (min(minimum(loss), chance)-10, chance*1.2+10)
            ps = map(dimnames(loss), axiskeys(loss), marginals) do name, x, l
                d = box[name]
                maybelog = :log in d ? (:log,) : ()
                # maybelog = ()
                plot(x, l; xaxis=(string(name), maybelog...), ylim, ylabel="Negative Log Likelihood")
                hline!([chance, chance*1.2], color=:gray, ls=:dash)
            end
            plot(ps..., size=(600, 600))
        end
    end
end

# plot_marginals("addm/grid/v9")
plot_marginals("bddm/grid/v2")
