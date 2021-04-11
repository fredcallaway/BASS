include("utils.jl")
include("model.jl")
include("dc.jl")
include("data.jl")
include("likelihood.jl")

using Test
using Random

# %% --------
# currently we just make sure it doesn't crash...

all_data = load_human_data()
trials = prepare_trials(all_data; dt=.025)
ibs_loglike(m, trials[1:50]; ε=.1)
