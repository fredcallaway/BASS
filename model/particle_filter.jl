using DataStructures
using StatsFuns
function fixation_vector(t::HumanTrial)
    attended = zeros(Int, t.rt)
    switch = make_switches(t)
    i = 1
    first_fix = -1
    while i < t.rt
        item, time_to_switch = switch()
        if first_fix == -1
            first_fix = time_to_switch
        end
        for j in 1:time_to_switch
            attended[i] = item
            i += 1
        end
    end
    attended, first_fix
end

struct ParticleFilter
    m::BDDM
    pol::Policy
    t::HumanTrial
    fix::Vector{Int}
    first_duration::Int
    objective_precision::Vector{Float64}
    attention::Vector{Float64}
    λ_objective::Vector{Float64}
end

function ParticleFilter(m::BDDM, pol::Policy, t::HumanTrial)
    pol = deepcopy(pol)
    initialize!(pol, t)
    fix, first_duration = fixation_vector(t)
    ParticleFilter(m, pol, t, fix, first_duration, base_precision(m, t), zeros(m.N), zeros(m.N))
end

mutable struct Particle
    splits::Int  # weight = 0.5 ^ splits
    s::State
    time_step::Int
end
Particle(pf::ParticleFilter) = Particle(0, State(m), 1)

function transition!(pf::ParticleFilter, ps::Particle)
    @unpack m, pol, t, fix, first_duration, objective_precision, attention, λ_objective = pf
    s = ps.s
    ps.time_step += 1
    set_attention!(attention, m, fix[ps.time_step], ps.time_step <= first_duration)
    @. λ_objective = objective_precision * attention
    update!(m, s, t.value, λ_objective, λ_objective)  # λ_subjective
    return ps
end

function run!(pf::ParticleFilter, n_particle=1000, callback=(particles->nothing))
    @unpack m, pol, t = pf

    alive = Queue{Particle}()
    for i in 1:n_particle
        enqueue!(alive, Particle(pf))
    end
    dead = Queue{Particle}()

    for time_step in 1:t.rt-1
        callback(alive)
        # group particles into living and dead
        for j in 1:n_particle
            ps = dequeue!(alive)
            @assert ps.time_step == time_step
            dest = stop(pol, ps.s, t) ? dead : alive
            enqueue!(dest, ps)
        end
        # split living particles to replace dead ones
        while !isempty(dead)
            d = dequeue!(dead)
            a = dequeue!(alive)
            d.s = copy(a.s)
            a.splits += 1
            d.splits = a.splits
            enqueue!(alive, a)
            enqueue!(alive, d)
            # putting them back at the end keeps the queue sorted
            # which means we minimize variance in the weights
        end
        for ps in alive
            transition!(pf, ps)
        end
    end
    callback(alive)
    # final time step: must stop and choose the correct item
    log_2 = log(2)
    logps = map(alive) do ps
        @assert ps.time_step == t.rt
        if stop(pol, ps.s, t) && argmax(subjective_values(m, ps.s)) == t.choice
            -log_2 * ps.splits
        else
            -Inf
        end
    end
    logsumexp(logps) - log(n_particle)
end

function run!(callback::Function, pf::ParticleFilter, n_particle=1000)
    run!(pf, n_particle, callback)
end
