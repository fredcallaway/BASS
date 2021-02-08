using Distributions

mutable struct IBSEstimate{F}
    sample_hit::F
    k::Int
    logp::Float64
end
IBSEstimate(f::Function) = IBSEstimate(f, 1, 0.)

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
    return (logp=total_logp / repeats, converged=true)
end

function ibs(sample_hit::Function, data::Vector; kws...)
    hit_samplers = map(data) do d
        () -> sample_hit(d)
    end
    ibs(hit_samplers; kws...)
end

# # %% --------

# truth = Bernoulli(0.5)
# N = 100
# data = [rand(truth) for i in 1:N]

# ps = 0.05:0.05:0.95
# res = map(ps) do p
#     # hit_samplers = map(data) do d
#     #     () -> rand(Bernoulli(p)) == d
#     # end
#     # ibs(hit_samplers; repeats=1000)
#     ibs(data; repeats=100) do d
#         rand(Bernoulli(p)) == d
#     end
# end

# using SplitApplyCombine
# lp, converged = invert(res)
# figure() do
#     plot(ps, lp)
# end


