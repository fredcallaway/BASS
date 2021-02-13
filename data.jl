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
    dists = [Normal(.2, .05), Normal(.5, .1)]
    avg_first = mean(durations[1:2:end])
    avg_second = mean(durations[2:2:end])
    avg_first < avg_second ? dists : reverse(dists)
end

function HumanTrial(d::NamedTuple; μ, σ, dt)
    presentation_times = get_presentation_dists(d.presentation_duration, dt)
    real_presentation_times = round.(Int, d.presentation_duration ./ dt)
    rt = sum(real_presentation_times)
    HumanTrial((d.value .- μ) ./ σ, d.confidence, presentation_times, 
               real_presentation_times, d.subject, d.choice, rt, dt)
end

function prepare_trials(data; dt=.01, normalize_value=true)
    μ, σ = normalize_value ? juxt(mean, std)(flatten(data.value)) : (0, 1)
    trials = map(data) do d
        HumanTrial(d; μ, σ, dt)
    end
    filter!(trials) do t
        # this can happen due to rounding error
        t.rt <= max_rt(t)
    end
end

# %% --------

function table(trials::Vector{HumanTrial})
    map(trials) do t
        x = ntfromstruct(t)
        m1, m2 = mean.(x.presentation_times)
        order = m1 > m2 ? :longfirst : :shortfirst
        presentation_duration = x.real_presentation_times .* t.dt
        total_presentation = [sum(presentation_duration[1:2:end]), sum(presentation_duration[2:2:end])]
        (;x.subject, x.value, x.confidence, presentation_duration, order, x.choice, rt=x.rt * t.dt)
    end |> Table
end

