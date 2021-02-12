using SpecialFunctions: trigamma

mutable struct IBSEstimate{F}
    sample_hit::F
    k::Int
    logp::Float64
end
IBSEstimate(f::Function) = IBSEstimate(f, 1, 0.)

Distributions.var(est::IBSEstimate) = trigamma(1) - trigamma(est.k)
Distributions.mean(est::IBSEstimate) = est.logp


function sample_hit!(est::IBSEstimate)
    if est.sample_hit()
        true
    else
        est.logp -= 1 / (est.k)
        est.k += 1
        false
    end
end

function ibs(hit_samplers::Vector{<:Function}; repeats=1, min_logp=-Inf)
    total_logp = 0.
    for i in 1:repeats
        unconverged = Set(IBSEstimate(f) for f in hit_samplers)
        converged_logp = 0.
        while !isempty(unconverged)
            unconverged_logp = 0.
            for est in unconverged
                if sample_hit!(est)
                    converged_logp += est.logp
                    delete!(unconverged, est)
                else
                    unconverged_logp += est.logp
                end
            end
            if converged_logp + unconverged_logp < min_logp
                return (logp=min_logp, converged=false)
            end
        end
        total_logp += converged_logp
    end
    nll_var = var(est)

    return (nll=-total_logp / repeats, converged=true)
end

function ibs(sample_hit::Function, data::Vector; kws...)
    hit_samplers = map(data) do d
        () -> sample_hit(d)
    end
    ibs(hit_samplers; kws...)
end

