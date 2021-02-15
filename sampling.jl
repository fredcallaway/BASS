
# %% ==================== Plot "samples" ====================
top = partialsortperm(nll, 1:100)
samples = map(pairs(box.dims), xs[top] |> invert) do (name, d), x
    name => x
end |> Dict

params = collect(keys(box.dims))

# %% --------
figure()
    plot_grid(params, params) do vx, vy
        scatter(samples[vx], samples[vy])
        plot!(xlim=(0,1), ylim=(0,1), xlabel=string(vx), ylabel=string(vy))
    end
end

# %% --------
figure() do
    vx = :base_precision
    vy = :confidence_slope
    scatter(samples[vx], samples[vy], smooth=true)
    plot!(xlim=(0,1), ylim=(0,1), xlabel=string(vx), ylabel=string(vy))

end

# %% --------
using DataFrames
using RCall
df = DataFrame(samples)

# %% --------
R"""
round(cor($df), 2)
"""
