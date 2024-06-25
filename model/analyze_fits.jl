include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("likelihood.jl")
include("ibs.jl")
include("box.jl")
using Serialization
using ProgressMeter
using Sobol
using SplitApplyCombine

@everywhere include("gp_likelihood.jl")

# %% ==================== GP likelihood ====================

processed = process_sobol_result(BDDM, "apr14", "group"; verbose=true)

# %% --------
full_res = deserialize("tmp/bddm/sobol/apr14/group");
logp = getfield.(full_res.results, :logp)
rank = sortperm(logp; rev=true)
logp[rank]
full_res.models[rank[1:10]]
full_res.models[rank[1]]

sres.trials

β = 1.
ε = .01

sres.trials[end]
data2 = load_human_data(2)
length(data2)
length(sres.trials)

sum(sres.trials) do t
    log(ε * 0.5 + (1 - ε) * softmax(β * t.value)[t.choice])
end

res.model_nll_true
BDDM(;sres.box(res.best_x)...)

res



# %% ==================== old ====================


version = "choice_only"
run_name = "bddm/sobol/$version/"
subjects = readdir("tmp/$run_name")
@showprogress "Processing sobol results" pmap(subjects) do subject
    process_sobol_result(BDDM, "$version", subject)
end


# %% --------

include("base.jl")
model = BDDM(;sres.prm...)
data1 = load_human_data(1)
df = write_sim(model, data1, "apr2", "1-fit")



order = sortperm(res.results, by=x->x.logp; rev=true)

# %% --------


m1_main = BDDM(
    base_precision = .05,
    attention_factor = 0.8,
    cost = .06,
    prior_mean = µ,
    prior_precision = 1 / σ^2,
)

# %% --------


# # %% --------

# include("base.jl")

# µ, σ = empirical_prior(data1)
# m1_main = BDDM(
#     base_precision = .05,
#     attention_factor = 0.8,
#     cost = .06,
#     prior_mean = µ,
#     prior_precision = 1 / σ^2,
# )
# ibs_loglike(m1_main, trials, ε=.05, tol=0, repeats=10, min_multiplier=1.2)
# chance_loglike(trials; tol=0)
# ibs_kws = (;ε, tol, repeats, min_multiplier)


# run_sobol_group(BDDM, "v11", box, 5000, repeats=10, dt=.025, tol=0)
# run_sobol_group(BDDM, "v12", box, 5000, repeats=10, dt=.025, tol=0)
# run_sobol_ind(BDDM, "v13", box, 5000, repeats=10, dt=.025, tol=0, data=load_human_data(2))