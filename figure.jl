using Plots
using Dates
mkpath("fighist")
mkpath("figs")
gr(label="", dpi=200, size=(400,300), lw=2)
ENV["GKSwstype"] = "nul"

function figure(f, name="tmp"; kws...)
    plot(;kws...)
    f()
    dt = Dates.format(now(), "m-d-H-M-S")
    p = "fighist/$dt-$name.png"
    savefig(p)
    if name != "tmp"
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
