X = reduce(hcat, xs)
gp_mean = MeanConst(gp_mean)
kernel = SEArd(ones(size(X, 1)), -3.)

log_noise = @.(log(√(nll_var)))
# good = findall(nll .< 862)
good = partialsortperm(nll, 1:3000)

# order = sortperm(nll)
# filter!(order) do i
#     nll_var[i] < 20
# end

train = good[1:2:end]
test = good[2:2:end]
gp_mean = MeanConst(ibs_kws.min_multiplier * -chance)

train_small = train[1:2:end]
kernel = SEArd(ones(size(X, 1)), -3.)
@time gp = GPE(X[:, train_small], nll[train_small], gp_mean, kernel, mean(log_noise[train_small]))
@time optimize!(gp, domean=false, noise=false)


# first fit on half the data to estimate out of sample prediction accuracy
@time gp = GPE(X[:, train], nll[train], gp_mean, kernel, mean(log_noise[train]))
@time optimize!(gp, domean=false, noise=false)

# %% --------
yhat, yvar = predict_f(gp, X[:, test])
rmse(preds) = √ mean((preds .- nll[test]).^2)
# mae(preds) = mean(abs.(preds .- nll[test]))
println("RMSE on held out data: ", rmse(yhat) |> round2)
println("RMSE of mean prediction: ", rmse(mean(nll[train])) |> round2)
println("RMSE of perfect prediction: ", √mean(nll_var) |> round2)
println("RMSE scaled by prediction variance: ", 
    mean(((yhat .- nll[test]) ./ .√yvar).^2) |> round2)
# %% --------
# use all the data for further analysis
kernel
gp = GPE(X, nll, gp_mean, kernel, log_noise)
optimize!(gp, domean=false, noise=false)
gp


# %% ==================== Sparse approx ====================
function assign_block(x; n=2)
    mapreduce(+, enumerate(x)) do (i, x)
        bin = min(n-1, Int(fld(x, 1/n)))
        1 + bin * n ^ (i - 1)
    end
end

function make_block_indices(X)
    blocks = [Int[] for i in 1:3^5]
    for (i, x) in enumerate(eachcol(X))
        push!(blocks[assign_block(x)], i)
    end
    blocks
end

using LinearAlgebra
gp = GaussianProcesses.FSA(X[:, train], Xu, make_block_indices(X[:, train]), nll[train],
                           gp_mean, kernel, log(mean(nll_std[train])))
@time yhat, yvar = predict_f(gp, X[:, test], make_block_indices(X[:, test]); full_cov=true)
yvar = diag(yvar)

# %% ==================== testing ====================


gp_mean = MeanConst(mean(nll))
kernel = SEArd(ones(size(X, 1)), -3.)

# splt = cld(length(nll), 2)
# train = eachindex(nll)[1:splt]
# test = eachindex(nll)[splt:end]
train = 1:2:length(nll)
test = 2:2:length(nll)

gp = GPE(X[:, train[1:500]], nll[train[1:500]], gp_mean, kernel, log.(nll_std[train]))
@time optimize!(gp, domean=false, noise=false)

gp = GPE(X[:, train], nll[train], gp_mean, gp.kernel, log.(nll_std[train]))
@time yhat, yvar = predict_f(gp, X[:, test])

rmse(preds) = √ mean((preds .- nll[test]).^2)
begin
    println("RMSE on held out data: ", rmse(yhat) |> round2)
    println("RMSE of mean prediction: ", rmse(mean(nll[train])) |> round2)
    println("RMSE of perfect prediction: ", √mean(nll_std) |> round2)
    println("RMSE scaled by prediction variance: ", 
        mean(((yhat .- nll[test]) ./ .√yvar).^2) |> round2)
end
# %% --------

order = sortperm(sr.nll)
filter!(order) do i
    sr.nll_std[i] < 20
end
# to_fit = [order[1:1000]; order[1001:10:end]]
to_fit = order[1:1000]

