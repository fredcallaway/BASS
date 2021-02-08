using JSON
using TypedTables
using SplitApplyCombine

function parse_indpres(x)
    if x isa Real
        return [x]
    elseif x[1] isa Real
        @assert x[1] == 0
        @assert length(x) == 2
        return [x[2]]
    else
        x1, x2 = x
        @assert length(x) == 2
        # @assert x1 isa Vector && x2 isa Vector
        if !(x1 isa Vector && x2 isa Vector)
            @show x1 x2
            error()
        end
        return x1 .+ x2
    end
end

# %% --------
all_data = map(open(JSON.parse, "data/Study3Fred.json")) do d
    presentation_duration = parse_indpres(d["IndPresDur"])
    avg_first = mean(presentation_duration[1:2:end])
    avg_second = mean(presentation_duration[2:2:end])
    (
        subject = d["SubNum"],
        value = [d["fstItemV"], d["sndItemVal"]],
        confidence = Float64[d["fstConfidence"], d["sndConfidence"]],
        presentation_duration,
        nfix = length(presentation_duration),
        order = avg_first > avg_second ? :longfirst : :shortfirst,
        choice = d["isFirstChosen"] ? 1 : 2,
        rt = d["RT"]
    )
end  |> skipmissing |> collect |> Vector{NamedTuple} |>  Table
# %% --------
function get_presentation_dists(durations, dt)
    dists = [Normal(.2/dt, .05/dt), Normal(.5/dt, .1/dt)]
    avg_first = mean(durations[1:2:end])
    avg_second = mean(durations[2:2:end])
    avg_first < avg_second ? dists : reverse(dists)
end

function HumanTrial(d::NamedTuple; μ, σ, dt)
    presentation_times = get_presentation_dists(d.presentation_duration, dt)
    real_presentation_times = round.(Int, d.presentation_duration ./ dt)
    rt = sum(real_presentation_times)
    HumanTrial((d.value .- μ) ./ σ, d.confidence, presentation_times, 
               real_presentation_times, d.subject, d.choice, rt)
end

function prepare_trials(data; dt=.01)
    μ, σ = juxt(mean, std)(flatten(data.value))
    map(data) do d
        HumanTrial(d; μ, σ, dt)
    end
end

# %% --------

