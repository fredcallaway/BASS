using Distributions
include("ibs.jl")
# %% --------


N = 100000
ps = 0.05:0.05:0.95

map(ps) do true_p
    truth = Bernoulli(true_p)
    data = [rand(truth) for i in 1:N]

    nll = map(ps) do p
        # hit_samplers = map(data) do d
        #     () -> rand(Bernoulli(p)) == d
        # end
        # ibs(hit_samplers; repeats=1000)
        res = ibs(data; repeats=1) do d
            rand(Bernoulli(p)) == d
        end
        -res.logp
    end
    @assert ps[argmin(nll)] == true_p
end

# %% --------
N = 1000
true_p = 0.5
truth = Bernoulli(true_p)
data = [rand(truth) for i in 1:N]
p = 0.5

ests = map(1:1000) do i
    ibs(data; repeats=4) do d
        rand(Bernoulli(p)) == d
    end
end

using SplitApplyCombine
lp, v = invert(ests)

var(lp)
mean(v)