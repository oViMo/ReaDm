using RaeDm, Test, Distributions

nt = 100 # trials
ns = 10 # subjects
RT = [rand(nt) for s in 1:ns]# reaction times
C = [rand([-1,1],nt) for s in 1:ns] # choices, 1 or 0
X = [[randn(2) for t in 1:nt] for s in 1:ns] # regressors

F = RaeDm.FIVOChain()

F(RT[1],C[1],X[1])
