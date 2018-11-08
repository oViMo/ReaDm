module SSM
using Distributions,  ForwardDiff , StatsFuns , Flux
export DDM,DDM2,DDM_lpdf_notape , ddm_avg

abstract type DDMdistributionC <: Distributions.ContinuousUnivariateDistribution end
abstract type DDMdistribution <: Distributions.ContinuousMultivariateDistribution end
struct DDM{Boundaries<:Real,Drift<:Real,StartPoint<:Real,nonDecisionTime<:Real} <: DDMdistribution
    a::Boundaries
    v::Drift
    w::StartPoint
    t0::nonDecisionTime
end
struct DDM2{Boundaries<:Real,Drift<:Real,StartPoint<:Real,nonDecisionTime<:Real,C<:Real} <: DDMdistributionC
    a::Boundaries
    v::Drift
    w::StartPoint
    t0::nonDecisionTime
    c::C
end
function DDM(a::T,v::T,w::T,t0::T) where T<:Real
DDM{typeof.([a,v,w,t0])...}(a,v,w,t0)
end
function DDM(a::T,v::T,w::T,t0::T,c::T) where T<:Real
DDM2{typeof.([a,v,w,t0,c])...}(a,v,w,t0,c)
end
function Distributions.length(d::DDMdistribution)
return 2
end
function Distributions._logpdf(d::DDMdistribution,x::AbstractArray) 
return DDM_lpdf(d.a,d.v,d.w,d.t0,x[1],x[2])
end
function Distributions._logpdf(d::DDMdistributionC,x::AbstractArray) 
return DDM_lpdf(d.a,d.v,d.w,d.t0,x[1],d.c)
end
function Distributions._logcdf!(r,d::DDMdistribution,x::AbstractArray)
#return r.=log(Fs(d.a,d.v,d.w,d.t0,x[1],x[2]))
return r.=log(DDM_cdf(d.a,d.v,d.w,d.t0,x[1],x[2]))
end
function Distributions._logcdf!(r,d::DDMdistributionC,x::AbstractArray)
#return r.=log(Fs(d.a,d.v,d.w,d.t0,x[1],d.c))
return r.=log(DDM_cdf(d.a,d.v,d.w,d.t0,x[1],d.c))
end
function Distributions.logcdf(d::DDMdistribution,x::AbstractArray)
#return log(Fs(d.a,d.v,d.w,d.t0,x[1],x[2]))
return log(DDM_cdf(d.a,d.v,d.w,d.t0,x[1],x[2]))
end
function Distributions.logcdf(d::DDMdistributionC,x::AbstractArray)
#return log(Fs(d.a,d.v,d.w,d.t0,x[1],d.c))
return log(DDM_cdf(d.a,d.v,d.w,d.t0,x[1],d.c))
end
function Distributions.cdf(d::DDMdistribution,x::AbstractArray)
#return Fs(d.a,d.v,d.w,d.t0,x[1],x[2])
return DDM_cdf(d.a,d.v,d.w,d.t0,x[1],x[2])
end
function Distributions.cdf(d::DDMdistributionC,x::AbstractArray)
#return Fs(d.a,d.v,d.w,d.t0,x[1],d.c)
return DDM_cdf(d.a,d.v,d.w,d.t0,x[1],d.c)
end
function Distributions._rand!(d::DDMdistribution,x::AbstractArray)
DDM_rand!(x,d.a,d.v,d.w,d.t0)
end


function Distributions.rand(d::DDMdistributionC)
c = 0.
x = 0.
X=zeros(2)
while c != d.c
	x,c = DDM_rand!(X,d.a,d.v,d.w,d.t0)
end
return x
end
function Distributions.mean(d::DDMdistribution)
return [ddm_avgrt(d.a,d.v,d.w,d.t0);ddm_avg(d.a,d.v,d.w)]
end
function Distributions.mean(d::DDMdistributionC)
return ddm_avgrt(d.a,d.v,d.w,d.t0,d.c)
end
function Distributions.var(d::DDMdistribution)
p1 = ddm_avg(d.a,d.v,d.w)
return [ddm_varrt(d.a,d.v,d.w,d.t0);p1*(1-p1)]
end
function Distributions.var(d::DDMdistributionC)
return ddm_varrt(d.a,d.v,d.w,d.t0,d.c)
end



const DDM_logistic(x) = StatsFuns.logistic(x)
const DDM_mapa = StatsFuns.softplus
import Base.floatmax
floatmax(x::Union{Type{ForwardDiff.Dual{X,T,V}},Type{Flux.Tracker.TrackedReal{T}}}) where {T,X,V} = floatmax(T)*one(x)
const logmax(x) = x>zero(x) ? log(x) : -floatmax(typeof(x))

function DDM_cdf(a,v,w,t0,rt,c)
@assert c==one(c) || c==-one(c)
v = -c*v
w = -c*w
a=DDM_mapa(a)
w=DDM_logistic(w)

rt = rt-t0
if rt>0
T0=0.2397217965550664
n=8
m=4
aa = a*a
V = rt - T0 * aa<0 ? DDM_cdf_small(a,v,w,rt) : DDM_cdf_large(a,v,w,rt,n,c)
return minmax01(V,rm=true)
else
	return zero(a)
end
end
function DDM_cdf_small(a,v,w,t,eps=sqrt(eps()))
	K = Int(round(Ks(t, v, a, w, eps)))
	F = zero(a)
	sqt = sqrt(t)
	for k in K:-1:0
		rj = 2*k*a + a*w
		dj = -v*a*w - v*v*t/2 + StatsFuns.normlogpdf(rj/sqt)
		pos1 = dj + logMill((rj-v*t)/sqt)
		pos2 = dj + logMill((rj+v*t)/sqt)
		rj = (2*k+1)*a + a*(1-w)
		dj = -v*a*w - v*v*t/2 + StatsFuns.normlogpdf(rj/sqt)
		neg1 = dj + logMill((rj-v*t)/sqt) 
		neg2 = dj + logMill((rj+v*t)/sqt)
		F = exp(pos1) + exp(pos2) - exp(neg1) - exp(neg2) + F
	end
	return isfinite(F) ? F : zero(F)
end
function DDM_cdf_large(a,v,w,t,n,c)
	P = (1-exp(-2*v*a*(1-w)) )/ (exp(2v*a*w)-exp(-2*v*a*(1-w)))
	P = isfinite(P) ? P : (sign(v/c)==sign(-c) ? 1. : 0.0)

	V=zero(a)
for k in 1:n
V+=k*sin(pi*k*w)/(v^2+(k*pi/a)^2) * exp(-.5(k*pi/a)^2*t)
end
V = V*exp(-v*a*w-.5v^2*t - 2log(a)+1.8378770664093453)
return P - (isfinite(V) ? V : zero(V))
end
function DDM_lpdf(a,v::T,w,t0,rt,c) where {T}
	DDM_lpdf(a,v,w,t0,convert(T,rt),convert(T,c))
end
function DDM_lpdf(a::T,v::T,w::T,t0::T,rt::T,c::T) where {T}
@assert c==one(c) || c==-one(c)
v = -c*v
w = -c*w
a = DDM_mapa(a)
w = DDM_logistic(w)
rt = rt-t0
return  rt>0 ? (ddm_p0(a,v,w,rt) + ddm_p1(a,w,rt)) : (-floatmax(T))
end
#function DDM_lpdf_notape(a,v,w,t0,rt,c)
#v = -c*v
#w = -c*w
#a=DDM_mapa(a)
#w=DDM_logistic(w)
#rt = rt-t0
#return   ddm_p0(a,v,w,rt) + ddm_p1_notape(a,w,rt)
#end

function ddm_p0(a::T,v,w,t)::T where {X,Y,T<:Union{Float64,Tracker.TrackedReal{Float64},ForwardDiff.Dual{X,Float64,Y}}}
return -a*v*w-2log(a)-(v*v)*.5t
end
function ddm_p0(a::T,v,w,t)::T where {X,Y,T<:Union{Float32,Tracker.TrackedReal{Float32},ForwardDiff.Dual{X,Float32,Y}}}
return -a*v*w-2log(a)-(v*v)*0.5f0 * t
end
#function ddm_p1_notape(a,w,t)
#	#T0=0.2145535792511116
#	#n=6
#	#m=3
#	T0=0.2397217965550664
#	n=8
#	m=4
#	aa=a*a
#	if t - T0 * aa <0
#		ddm_logpdf_full_common1_GRAD(a,t,w,aa,m) 
#	else
#		ddm_logpdf_full_common2_GRAD(a,t,w,aa,n)
#	end
#end

function ddm_p1(a::T,w::T,t::T)::T where T
#T0=0.2145535792511116
#n=6
#m=3
T0=0.2397217965550664
n=8
m=4
aa=a*a
V=if t-T0*aa < 0
	ddm_logpdf_full_common1_GRAD(a,t,w,aa,m)
else
	ddm_logpdf_full_common2_GRAD(a,t,w,aa,n)
end
end

# Convert type
import Flux.Tracker: TrackedReal
Flux.Tracker.TrackedReal{Float32}(x::Float64) = Float32.(x)
Flux.Tracker.TrackedReal{Float32}(x::Float32) = x
Flux.Tracker.TrackedReal{Float64}(x::Float64) = x
Flux.Tracker.TrackedReal{Float32}(x::Flux.Tracker.TrackedReal) = Float32.(x)
Flux.Tracker.TrackedReal{Float64}(x::Flux.Tracker.TrackedReal) = Float64.(x)

function ddm_logpdf_full_common1_GRAD(a::T,t::T,w::T,aa::T,m::Int64)::T where {X,Y,T<:Union{Float64,Tracker.TrackedReal{Float64},ForwardDiff.Dual{X,Float64,Y}}}
Ot    = 1.0/t
Ot_aa = Ot*aa
cst   = .5Ot_aa*w*w

V=zero(T)
for k2=-m:m
    TKW = 2k2+w
    V += TKW*exp(-.5Ot_aa*TKW*TKW + cst)
end
return logmax(V) + 3log(a) - 1.5log(t) - 9.189385332046727e-1 - cst 
end
function ddm_logpdf_full_common1_GRAD(a::T,t::T,w::T,aa::T,m::Int64)::T where {X,Y,T<:Union{Float32,Tracker.TrackedReal{Float32},ForwardDiff.Dual{X,Float32,Y}}}
Ot    = 1.0f0/t
Ot_aa = Ot*aa
cst   = 0.5f0 * Ot_aa*w*w

V=zero(T)
for k2=-m:m
    TKW = 2k2+w
    V += TKW*exp(-0.5f0 * Ot_aa*TKW*TKW + cst)
end
return logmax(V) + 3log(a) - 1.5f0 * log(t) - 0.9189385f0 - cst 
#return T(logmax(V) + 3log(a) - 1.5log(t) - convert(T,9.189385332046727e-1) - cst )
end


function ddm_logpdf_full_common2_GRAD(a::T,t::T,w::T,aa::T,n::Int64)::T where {X,Y,T<:Union{Float64,Tracker.TrackedReal{Float64},ForwardDiff.Dual{X,Float64,Y}}}
# Navarro and Fuss 2009
cst = t*4.934802200544679/aa # Numerical underflow
V = zero(T)
for k=1:n
        V += sin(3.14159265358979*w*k)*k*exp(-cst*(k*k-1))
end
return logmax(V) - cst + 1.1447298858494
end
function ddm_logpdf_full_common2_GRAD(a::T,t::T,w::T,aa::T,n::Int64)::T where {X,Y,T<:Union{Float32,Tracker.TrackedReal{Float32},ForwardDiff.Dual{X,Float32,Y}}}
# Navarro and Fuss 2009
cst = t*4.934802f0/aa # Numerical underflow
V = zero(T)
for k=1:n
        V += sin(3.1415927f0*w*k)*k*exp(-cst*(k*k-1))
end
return logmax(V) - cst + 1.1447299f0
end

function ddm_avgrt(a,v,w,t0;s=1)
# Grasman 2009
a=DDM_mapa(a)
w = DDM_logistic(w)

z = w*a
Z = expm1(-2*v*z/s^2)
A = expm1(-2*a*v/s^2)
return t0 + -z/v + a/v * Z/A
end
function ddm_avgrt(a,v,w,t0,c)
# Grasman 2009

a=DDM_mapa(c*a)
w = DDM_logistic(c*w)

av = a*v 
avw = av*w
Q3=expm1(-2*av)
Q4=expm1(-2*avw)

    M0 = abs(v)>1e-3 ? a.*(Q4-Q3.*w)./(v.*Q3)       :  - a.^2.0 .* ((-1.0)+w).*w        

    return t0+M0 

end
function ddm_varrt(a,v,w,t0;s=1)
# Grasman 2009
a=DDM_mapa(a)
w = DDM_logistic(w)

z = w*a
Z = expm1(-2*v*z/s^2)
A = expm1(-2*a*v/s^2)
(-v*a^2 * (Z+4)*Z/A^2 + ((-3*v*a^2 + 4*v*z*a + s^2 * a)*Z + 4*v*z*a)/A - s^2 * z)/v^3
end
function ddm_varrt(a,v,w,t0,c)

a=DDM_mapa(c*a)
w = DDM_logistic(c*w)

av = a*v 
avw = av*w
Q3=expm1(-2*av)
Q4=expm1(-2*avw)

    V0 = abs(v)>1e-3 ? -a./(v.^3.0) .*(w-(Q4.*(a.*(w.*4.0-3.0).*v+1.0)+avw.*4.0)/Q3+Q4.*av.*(Q4+4.0)/Q3.^2.0) : .33333333333333 .*a.^4.0 .* w.* (1.0+w.*((-3.0)+(-2.0).*((-2.0)+w).*w))  

    return  V0

end
function ddm_avg(a,v::T,w,s::T=one(T),c::T=one(T)) where T
# limit of cdf for t->inf at lower barrier
w = -w*c
v = -v*c

w=DDM_logistic(w)
a=DDM_mapa(a)

v = v/s^2

V = (exp(-2v*a*w)-exp(-2*v*a))/(-expm1(-2*v*a))
V = isfinite(V) ? V : 1/(exp(2v*a*w)-expm1(-2*v*a*(1-w)))-1/(exp(2v*a)-1)
if zero(V)<=V<=one(V)
	V
else
	zero(V)
end

end
const normlogCDF(z) = z < -1.0 ?
    log(erfcx(-z * 0.7071067811865475)/2) - z^2/2 :
        log1p(-erfc(z * 0.7071067811865475)/2)
function logMill(x)
# Gondan et al. 2014
return x > 10000 ? -log(x) : normlogCDF(-x) - StatsFuns.normlogpdf(x)
end
quant(x) = Distributions.quantile(Normal(),x)
function Ks(t,v,a,w,eps)
    K1  = (x->isfinite(x) && abs(x)<1000 ?  x : sign(x)*1000)(ceil((abs(v)*t - a*w)/2/a))
    V = exp(v*a*w + v*v*t/2 + log(eps))/2
    arg = maximum([zero(V)+typemin(Float64), minimum([one(V)-typemin(Float64), V])])

    V = quant(arg)

    V = (-sqrt(t)/2/a) * V
	K2=(x->isfinite(x) && abs(x)<1000 ? x : sign(x)*1000)(V)
    return ceil(maximum([K1, K1 + K2]))
end
function Fs(a,v,w,t0,t,c,eps=sqrt(eps()))
        w = -w*c
        v = -v*c
        w = DDM_logistic(w)
        a=DDM_mapa(a)
        t = t-t0

	K = Int(Ks(t, v, a, w, eps))
	F = 0.
	sqt = sqrt(t)
	for k in K:-1:0
		rj = 2*k*a + a*w
		dj = -v*a*w - v*v*t/2 + StatsFuns.normlogpdf(rj/sqt)
		pos1 = dj + logMill((rj-v*t)/sqt)
		pos2 = dj + logMill((rj+v*t)/sqt)
		rj = (2*k+1)*a + a*(1-w)
		dj = -v*a*w - v*v*t/2 + StatsFuns.normlogpdf(rj/sqt)
		neg1 = dj + logMill((rj-v*t)/sqt) 
		neg2 = dj + logMill((rj+v*t)/sqt)
		F = exp(pos1) + exp(pos2) - exp(neg1) - exp(neg2) + F
	end
	return F
end
function ks(t,w,eps)
	K2 = (sqrt(2*t) - w)/2
        k1 = copy(K2)
	u_eps = minimum([-1, log(2*pi*t*t*eps*eps)]) # Safe bound so that
	arg = -t * (u_eps - sqrt(-2*u_eps - 2)) # sqrt(x) with x > 0
        if arg>0
		K2 = .5 * sqrt(arg) - w/2
	end
	return ceil(maximum([K1, K2]))
end
function fsw(t,w,eps)
    K = ks(t, w, eps)
    f = similar(t)
    if K > 0
        for k in K:-1:1
        f = (w+2*k) * exp(-(w+2*k) * (w+2*k)/2/t) +
            (w-2*k) * exp(-(w-2*k) * (w-2*k)/2/t) + f
        end
    end
    return 1/sqrt(2*pi*t*t*t) * (f + w * exp(-w*w/2/t))
end
function fs(t,v,a,w,eps=sqrt(eps()))
    g = 1/a/a * exp(-v*a*w - v*v*t/2)
    return g * fsw(t/a/a, w, eps/g)
end

const h=1e-6
const delta=sqrt(h)
function DDM_rand!(A::Array{T,1},a::T,v::T,w::T,t0::T) where T
#0.00223606797749979 :: Float64

tt=1::Int64
a::T=DDM_mapa(a)/delta
pos::T=a*DDM_logistic(w)

	
pdown::T=0.5*(1.0-v*delta) 

@inbounds while zero(T)<=pos<=a
    tt+=1
    pos::T+=pdown<rand() ? 1. : -1.
end
		A[1]=tt*h+t0
if pos>=a
	A[2]=1.0
elseif pos<=zero(pos)
	A[2]=-1.0
end
A
#Action = pos>=a ? 1.0 : -1.0

#RT=tt*h+t0
#return RT,Action
end
function DDM_rand2(a::T,v::T,w::T,t0::T,s::Float64=1.) where T
# Euler-Maruyama


	tt=1::Int64
	a=DDM_mapa(a)
	pos=a*DDM_logistic(w)

	dv = h*v 
	sdelta = s*delta
	while true
		tt+=1
		pos+=dv + sdelta * randn()
		if pos>=a
			return tt*h+t0,1.0
		elseif pos<=0.
			return tt*h+t0,-1.0
		end
	end
#Action = pos>=a ? 1.0 : -1.0
	#RT=tt*h+t0
	#return RT,Action

end


function randDDMdis(a,v,w,t0,s=1)
# this formulation is clearer but strictly equivalent to the previous one
       δ = 1e-7
       aD=DM_mapa(a)
       z = logistic(w)*a
       t = 0
       sδs=sqrt(δ)*s
       δv = δ*v
       while 0<z<a
           t+=1
           z+=δv + sδs*randn()
       end
       if z>a
       A=1.
       else
       A=-1.
       end
       return t0+t*δ,A
       end

function generateDDMn(n::Int32,a::Float64,v::Float64,w::Float64,t0::Float64)
    RT=zeros(2,n);
    tic();
    for k=1:n
    RT[:,k]=generateDDM(a,v,w,t0);
    end
    toc();
    return RT; 
end
@inline function minmax01(V;rm=false)
        if rm
                cst = typemin(Float64)
        else
                cst=0.
        end 
        minimum([one(V)-cst;maximum([zero(V)+cst;V])])
end

end
