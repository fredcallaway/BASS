using Plots, Plots.Measures
using Dates
mkpath("fighist")
mkpath("figs")
gr(label="", dpi=200, size=(400,300), lw=2)
ENV["GKSwstype"] = "nul"

function figure(f, name="tmp"; kws...)
    plot(;kws...)
    f()
    dt = Dates.format(now(), "m-d-H-M-S")

    p = "fighist/$dt-$(replace(name, "/" => "-")).png"
    savefig(p)
    if name != "tmp"
        mkpath(dirname("figs/$name"))
        cp(p, "figs/$name.png"; force=true)
    end
end

_watch_process = nothing

function start_watch()
    if _watch_process == nothing
        global _watch_process = run(`watch fighist`; wait=false)
        println("running `watch fighist`")
    else
        println("`watch fighist` already running")
    end
end

function kill_watch()
    if _watch_process == nothing
        println("`watch fighist` not running")
    else
        kill(_watch_process)
        global _watch_process = nothing
        println("killing `watch fighist`.")
    end
end

function toggle_watch()
    if _watch_process == nothing
        global _watch_process = run(`watch fighist`; wait=false)
        println("running `watch fighist`")
    else
        kill(_watch_process)
        global _watch_process = nothing
        println("killing `watch fighist`.")
    end
end


function Plots.heatmap(X::KeyedArray{<:Real,2}; kws...)
    ylabel, xlabel = dimnames(X)
    heatmap(reverse(axiskeys(X))..., X; xlabel, ylabel, kws...)
end

function Plots.plot(x::KeyedArray{<:Real,1}; kws...)
    plot(axiskeys(x, 1), collect(x); xlabel=string(dimnames(x, 1)), kws...)
end

function Plots.plot(X::KeyedArray{<:Real,2}; kws...)
    k = dimnames(X, 2)
    plot(axiskeys(X, 1), collect(X);
        xlabel=dimnames(X, 1),
        label=reshape(["$k=$v" for v in axiskeys(X, 2)], 1, :),
        palette=collect(cgrad(:viridis, size(X, 2), categorical = true)),
        kws...
    )
end

function plot_grid(f::Function, rows, cols; size=(300, 300))
    ps = map(Iterators.product(rows, cols)) do (r, c)
        p = f(r, c)
        p
    end
    nr, nc = map(length, (rows, cols))
    plot(ps..., size=size .* (nr, nc), layout=(nc,nr), bottom_margin=4mm)
end

