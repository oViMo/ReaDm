using StatsFuns
using Flux.Tracker: TrackedReal, track, @grad, data
function rtnuni(μ,σ,b)
        # truncated normal distribution sampler (upper bound b)
        tp=normcdf(μ,σ,b)
        β = (b-μ)/σ
	x = μ+σ*Distributions.randnt(-Inf,data(β),data(tp))#randn()
	if !(x<b)
		@show x,b
	end
	x
end

rtnuni(μ::Union{TrackedArray,TrackedReal},σ::Union{TrackedArray,TrackedReal},b::Union{Float64,TrackedArray,TrackedReal}) = Tracker.track(rtnuni,μ,σ,b)

@grad function rtnuni(μ,σ,b) # Knowles 2015
	μv = data(μ)
	sv = data(σ)
	bv = data(b)

        xout = rtnuni(μv,sv,bv)

        β = (bv-μv)/sv
        #Z = normcdf(β)
        logZ = normlogcdf(β)
        #∇Z = -normpdf(β)
        log∇Z = normlogpdf(β)
        r2 = (xout-μv)/sv
        #npratio = normcdf(r2)/normpdf(r2)
        lognpratio = normlogcdf(r2)-normlogpdf(r2)
	grad = zeros(3)
	expval = exp(log∇Z+lognpratio-logZ)
        for (i,aa,bb) in zip(1:3,(1.,r2,0.),(-1.,-β,1.))
		grad[i] = (aa+bb*expval)
        end
	return xout,Δ -> (Δ * grad[1],Δ * grad[2],Δ * grad[3])
end

function entropyTMN(μ,σ,b)
β = (b-μ)/σ
logZ = normlogcdf(β)
H = log(σ)+.5*(StatsFuns.log2π+1)+logZ-.5*β*exp(normlogpdf(β)-logZ)
-H
end

import StatsFuns: normlogcdf, normlogpdf
normlogpdf(a::TrackedReal) = Tracker.track(normlogpdf, a)
normlogcdf(a::TrackedReal) = Tracker.track(normlogcdf, a)

@grad function normlogcdf(a)
	xv = data(a)
	v = StatsFuns.normlogcdf(xv)
	∂ = exp(-.5((xv^2)+StatsFuns.log2π)-v)
	return v, Δ -> (Δ*∂,)
end
@grad function normlogpdf(a)
	xv = data(a)
	v = -.5((xv*xv)+StatsFuns.log2π)
	∂ = -xv
	return v, Δ -> (Δ*∂,)
end

