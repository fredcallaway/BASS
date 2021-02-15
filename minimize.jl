"Wrappers around optimize"

using Optim
using Sobol

function fillmissing(target, repls)
    @assert sum(ismissing.(target)) == length(repls)
    repls = Iterators.Stateful(repls)
    x = Array{Float64}(undef, length(target))
    for i in eachindex(target)
        if ismissing(target[i])
            x[i] = first(repls)
        else
            x[i] = target[i]
        end
    end
    x
end

function minimize(f::Function; restarts=20)
    best_val = Inf; best_x = nothing
    for x0 in Iterators.take(SobolSeq(n_free(box)), restarts)
        res = optimize(x0) do x
            f(x)
        end
        if res.minimum < best_val
            best_x = res.minimizer
            best_val = res.minimum
        end
    end
    best_x, best_val
end

function minimize(f::Function, condition::Vector{Union{T,Missing}}; restarts=20) where T
    best_val = Inf; best_x = nothing
    for x0 in Iterators.take(SobolSeq(sum(ismissing.(condition))), restarts)
        res = optimize(x0) do x_opt
            x = fillmissing(condition, x_opt)
            f(x)
        end
        if res.minimum < best_val
            best_x = res.minimizer
            best_val = res.minimum
        end
    end
    best_x, best_val
end
