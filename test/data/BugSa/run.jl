using Pkg, Flux
using Flux: Tracker

Pkg.activate("./")
using RaeDm
print("Pkg loaded!\n")
include("load.jl")

F = FIVOChain(nlayers=0,nx=length(X[1][1]),nsim=8)

Fg = gpu(F)

interval = 500
L = F(RT[1],C[1],X[1],gradient_fetch_interval=interval)
Tracker.back!(L)

zero_grad!(F)

@time begin
    L = F(RT[1],C[1],X[1],gradient_fetch_interval=interval)
    Tracker.back!(L)
    zero_grad!(F)
end

opt = RaeDm.optimize(Flux.ADAM(params(F), 0.001))
opt = opt(F,RT,C,X,gradient_fetch_interval=interval)



