mkpath("figs/grid/")

for subject in  readdir("tmp/grid/feb7")
    @unpack box, results, chance = deserialize("tmp/grid/feb7/" * subject)
    loss = -getfield.(results, :logp)
    chance = -chance

    marginals = map(dimnames(loss)) do d
        dims = setdiff(dimnames(loss), [d])
        minimum(loss; dims) |> dropdims(dims...)
    end

    figure("grid/" * subject) do
        ps = map(dimnames(loss), axiskeys(loss), marginals) do name, x, l
            d = box[name]
            maybelog = :log in d ? (:log,) : ()
            # maybelog = ()
            plot(x, l, xaxis=(string(name), maybelog...))
            hline!([chance], color=:gray, ls=:dash)
        end
        plot(ps..., size=(600, 600))
    end
end