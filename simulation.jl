function write_simulation(name, model, data; dt=.025)
    μ, σ = juxt(mean, std)(flatten(data.value))
    map(data) do d
        t = HumanTrial(d; μ, σ, dt)
        sim = simulate(m, SimTrial(t); save_presentation=true)
        presentation_duration = t.dt .* sim.presentation_durations


        val1, val2 = d.value
        conf1, conf2 = d.confidence
        pt1 = round(sum(d.presentation_duration[1:2:end]); digits=3)
        pt2 = round(sum(d.presentation_duration[2:2:end]); digits=3)
        (;val1, val2, conf1, conf2, pt1, pt2, sim.choice)
    end |> Table |> CSV.write("results/$name.csv")
end

name = "bddm-v11-group"
mle = deserialize("results/$name-mle")
m = BDDM(;mle.prm...)
dt = .025
write_simulation(name, m, load_human_data())
