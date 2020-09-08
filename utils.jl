function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function map_product(f, args...; kws...)
    ks = keys(kws)
    n_pos = length(args)
    map(Iterators.product(args..., values(kws)...)) do x
        pos_args = x[1:n_pos]; kw_args = x[n_pos+1:end]
        f(pos_args...; Dict(zip(ks, kw_args))...)
    end
end