const DEBUG = true
#const isCuda = false

using RaeDm, Flux
print("RaeDm loaded\n")
using Flux: Tracker


nt = 1000 # trials
ns = 10 # subjects
RT = [1.0f0 .+ rand(Float32, nt) for s in 1:ns]# reaction times
C = [rand(Float32[-1,1,0], nt) for s in 1:ns] # choices, 1 or 0
X = [[randn(Float32, 2) for t in 1:nt] for s in 1:ns] # regressors

print("Creating FIVO object\n")
F = RaeDm.FIVOChain(nsim=8,nlayers=0)
if RaeDm.isCuda[]
    Fg = gpu(F)
end


if RaeDm.isCuda[]
    print("Loss (gpu):\n")
    L = Fg(RT[1],C[1],X[1])
    @show L
    print("backprop (gpu)\n")
    Tracker.back!(L)
    zero_grad!(Fg)
end
print("Loss:\n")
L = F(RT[1],C[1],X[1])
@show L
print("backprop\n")
Tracker.back!(L)
zero_grad!(F)


print("Time:\n")
@time begin
    L = F(RT[2],C[2],X[2])
    Tracker.back!(L)
end

print("Gradients:\n")
for _p in params(F)
	@show sum(abs2, _p.tracker.grad)
end

zero_grad!(F)

if RaeDm.isCuda[]
    print("Time (gpu):\n")
    @time begin
        L = Fg(RT[2],C[2],X[2])
        Tracker.back!(L)
    end
    zero_grad!(Fg)
end

using Profile
@profile begin
    L = F(RT[2],C[2],X[2])
end
Profile.print(format=:flat)
zero_grad!(F)
