using Pkg, Flux
using Flux: Tracker

Pkg.activate("./")
using RaeDm
print("Pkg loaded!\n")
include("load.jl")

F = FIVOChain(nlayers=0,nx=length(X[1][1]),nsim=8)

Fg = gpu(F)

interval = 500
L = F(RT[1],C[1],X[1],gradient_fetch_interval=interval,compute_intermediate_grad=false)
Tracker.back!(L)
zero_grad!(F)

print("with compute_intermediate_grad on\n")
@time begin
    L = F(RT[1],C[1],X[1],gradient_fetch_interval=interval,compute_intermediate_grad=true)
end
print("with compute_intermediate_grad off\n")
@time begin
    L = F(RT[1],C[1],X[1],gradient_fetch_interval=interval,compute_intermediate_grad=false)
    Tracker.back!(L)
    zero_grad!(L)
end
print("single update\n")
@time begin
	L = F(RT[1],C[1],X[1],gradient_fetch_interval=interval,compute_intermediate_grad=true,single_update=true)
end
opt = RaeDm.optimize(Flux.ADAM(params(F), 0.00001))
opt = opt(F,RT,C,X,gradient_fetch_interval=interval,compute_intermediate_grad=true,single_update=true,continuous_opt=false)



