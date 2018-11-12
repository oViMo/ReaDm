using Pkg, Flux
using Flux: Tracker

Pkg.activate("./")
using RaeDm
print("Pkg loaded!\n")
include("load.jl")

F = FIVOChain(nlayers=0,nx=length(X[1][1]),nsim=8)

L = F(RT[1][1:100],C[1][1:100],X[1][1:100])
Tracker.back!(L)

zero_grad!(F)

@time begin
    L = F(RT[1],C[1],X[1])
    Tracker.back!(L)
end

optimize(F,RT,C,X)
