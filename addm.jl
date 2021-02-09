using Random
using Distributions
using Parameters
#=
f: fixated item
ft: a fixation time or a trial
v: a vector of item ratings/values
E: accumulated evidence
=#

@with_kw struct ADDM
    θ::Float64 = 0.3
    d::Float64 = .0002
    σ::Float64 = .02
end

# %% ==================== Binary Accumulation ====================


function simulate(m::ADDM, t::SimTrial; dt, max_t=100000, kws...)
    _simulate(m, t, dt, max_t; kws...)
end

function simulate(m::ADDM, t::HumanTrial; kws...)
    _simulate(m, t, t.dt, t.rt; kws...)
end

function _simulate(m::ADDM, t::Trial, dt::Float64, max_t::Int;
                  save_history = false, save_fixations = false)
    @unpack θ, d, σ = m
    d *= (dt / .001)  # rescale the parameters to account for different time step size
    σ *=  √(dt / .001)  # this makes the predictions roughly insensitive to dt
    
    v = t.value
    @assert length(v) == 2
    switch = make_switches(t)
    noise = Normal(0, σ)
    history = save_history ? Float64[] : nothing
    E = 0.  # total accumulated evidence
    xx = d .* [v[1] - θ * v[2], θ * v[1] - v[2]]  # the two accumulation rates
    choice = -1
    ft = 0  # no initial fixation

    fix_times = save_fixations ? Int[] : nothing

    x = NaN  # have to initialize outside the if statement
    timeout = true
    rt = 0
    while rt < max_t
        rt += 1
        ε = rand(noise)
        if ft == 0
            f, ft = switch()
            save_fixations && push!(fix_times, ft)
            x = xx[f]
        end
        E += x + ε
        ft -= 1
        save_history && push!(history, copy(E))

        if !(-1 < E < 1)
            choice = E > 1 ? 1 : 2
            timeout = false
            if save_fixations
                fix_times[end] -= ft  # remaining fixation time
            end
            break
        end
    end
    (;choice, rt, timeout, fix_times, history)
end

