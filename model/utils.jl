using DataStructures: OrderedDict
using NamedTupleTools
using Statistics
using StatsBase
using Distributions
using SplitApplyCombine
flatten = SplitApplyCombine.flatten

macro infiltry(ex)
    return quote
        try
            $(esc(ex))
        catch
            $(Infiltrator.start_prompt)($(__module__), Base.@locals, $(String(__source__.file)), $(__source__.line))
        end
    end
end

named_tuple(x::NamedTuple) = x
function named_tuple(x)
    n = Tuple(propertynames(x))
    NamedTuple{n}(getproperty.(Ref(x), n))
end

Base.get(name::Symbol) = Base.Fix2(getproperty, name)
Base.get(i::Int) = Base.Fix2(getindex, i)
Base.get(x, name::Symbol) = getproperty(x, name)
Base.get(x, i::Int) = getindex(x, i)
Base.getindex(key) = Base.Fix2(getindex, key)

basetype(f::Type) = f.name.wrapper
basetype(f) = basetype(typeof(f))

flatmap(f, xs...; kws...) = mapreduce(f, vcat, xs...; kws...)

function monte_carlo(f, N=10000)
    N \ mapreduce(+, 1:N) do i
        f()
    end
end

function repeatedly(f, N=10000)
    map(1:N) do i
        f()
    end
end

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

call(f) = f()

function logspace(low, high, n)
    x = range(0, 1, length=n)
    @. exp(log(low) + x * (log(high) - log(low)))
end

function softmax!(x)
    x .= exp.(x .- maximum(x))
    x ./= sum(x)
end
softmax(x) = softmax!(copy(x))

function softmax!(x, i)
    x .= exp.(x .- maximum(x))
    x[i] / sum(x)
end

dictkeys(d::Dict) = (collect(keys(d))...,)
dictvalues(d::Dict) = (collect(values(d))...,)

round1(x) = round(x; digits=1)
round2(x) = round(x; digits=2)
round3(x) = round(x; digits=3)
round4(x) = round(x; digits=4)

fillmissing(x, rep) = ismissing(x) ? rep : x
fillmissing(rep) = x-> ismissing(x) ? rep : x

# namedtuple(d::Dict{String,T}) where {T} =
#     NamedTuple{Symbol.(dictkeys(d))}(dictvalues(d))
# namedtuple(d::Dict{Symbol,T}) where {T} =
#     NamedTuple{dictkeys(d)}(dictvalues(d))
# namedtuple(x) = namedtuple(Dict(fn => getfield(x, fn) for fn in fieldnames(typeof(x))))

Base.map(f::Function) = xs -> map(f, xs)
Base.map(f::Type) = xs -> map(f, xs)
Base.map(f, d::Dict) = [f(k, v) for (k, v) in d]

Base.filter(f::Function) = xs -> filter(f, xs)

Base.dropdims(idx::Int...) = X -> dropdims(X, dims=idx)
Base.reshape(idx::Union{Int,Colon}...) = x -> reshape(x, idx...)


valmap(f, d::AbstractDict) = Dict(k => f(v) for (k, v) in d)
valmap(f, d::OrderedDict) = OrderedDict(k => f(v) for (k, v) in d)
# valmap(f, d::T) where T <: AbstractDict = T(k => f(v) for (k, v) in d)

valmap(f) = d->valmap(f, d)
keymap(f, d::Dict) = Dict(f(k) => v for (k, v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)

function Base.write(fn)
    obj -> open(fn, "w") do f
        write(f, string(obj))
    end
end

function writev(fn)
    x -> begin
        write(fn, x)
        run(`du -h $fn`)
    end
end

# type2dict(x::T) where T = Dict(fn=>getfield(x, fn) for fn in fieldnames(T))

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))

# %% ==================== Axis Keys ====================
using AxisKeys
function grid(;kws...)
    vals = collect(values(kws))
    has_multi = map(vals) do v
        applicable(length, v) && length(v) > 1
    end
    singles = (;zip(keys(kws)[.!has_multi], vals[.!has_multi])...)
    multis = (;zip(keys(kws)[has_multi], vals[has_multi])...)
    X = map(Iterators.product(values(multis)...)) do x
        nt = (; singles..., zip(keys(multis), x)..., )
        NamedTuple(k => nt[k] for k in keys(kws))
    end
    KeyedArray(X; multis...)
end

function keyed(name, xs)
    KeyedArray(xs; Dict(name => xs)...)
end

keymax(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmax(X).I))...)
keymax(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmax(x)]

keymin(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmin(X).I))...)
keymin(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmin(x)]

Base.dropdims(idx::Union{Symbol,Int}...) = X -> dropdims(X, dims=idx)


function table(X::KeyedArray)
    map(collect(pairs(X))) do (idx, v)
        keyvals = (name => keys[i] for (name, keys, i) in zip(dimnames(X), axiskeys(X), idx.I))
        (;keyvals..., value=v)
    end[:]
end

function table(X::KeyedArray{<:NamedTuple})
    map(collect(pairs(X))) do (idx, v)
        keyvals = (name => keys[i] for (name, keys, i) in zip(dimnames(X), axiskeys(X), idx.I))
        (;keyvals..., v...,)
    end[:]
end

