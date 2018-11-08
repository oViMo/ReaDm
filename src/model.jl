#@mainDef afun = elu
const afun = elu

## Cuda ##
#========#
if isCu
	using CuArrays
	using CUDAnative
	#import Base: exp,log1p
	#import StatsFuns:log1pexp,softplus
	const MainType = Float32
	function send_effector(x::AbstractArray)
		isempty(x) ? x : (x |> gpu)
	end
	function send_effector(x)
		x
	end
	function send_effector(x::T) where {T<:Union{FIVOChain,Flux.Chain,Flux.Dense}}
		mapleaves(cu,x)
	end
	function send_effector(x::T) where {T<:Union{AbstractArray{<:AbstractArray},Tuple,NamedTuple}}
		map(send_effector,x)
	end
else
	const MainType = Float64
	function send_effector(x)
		x
	end
end

## Exec ##
#========#
function (fc::FIVOChain)(RT,C,x)
	if isCu
		RT = MainType.(RT)
		C = MainType.(C)
		x = cu.(x)
	end
	nsim	= fc.nsim
	nnodes  = fc.nnodes
	nlayers = fc.nlayers
	nx	= fc.nx
	ny	= fc.ny
	nz	= fc.nz


	ntrials = length(RT)
	accumulated_logw = param(zeros(MainType,1,nsim))
	L = zero(MainType)

	XYZ = [send_effector(param(zeros(MainType,nnodes,nsim))) for k in 1:3]
	for (rt,c,k) in zip(RT,C,1:ntrials)
		Yt      = fc.xPX(x[k]) # latent representation of regressors
		h	= fc.G(vcat(XYZ...)) # map previous regressors, data and latent variable to hidden state of GRU
		XYZ[2] = repeat(Yt,outer=(1,nsim))
		D = [rt;c]
		log_alpha_t = local_lik(h,fc,XYZ,D,nz)
		log_p_hat_t_summand = log_alpha_t .+ accumulated_logw
		log_p_hat_t = logsumexp_overflow(log_p_hat_t_summand)
		L = L + log_p_hat_t
		accumulated_logw = log_p_hat_t_summand - log_p_hat_t
		accumulated_logw,h,XYZ[3] = resample(accumulated_logw,h,XYZ[3])
	end
	L/ntrials
end
function resample(accumulated_logw,h,z)
acc_logw_detach = copy(accumulated_logw.data)
N = length(acc_logw_detach)
if -logsumexp_overflow(2 * acc_logw_detach) < log(N)-log(2)
	a = findfirst(cumsum(exp.(acc_logw_detach)) .> rand())
	accumulated_logw = -log(N)
	z = repmat(z[:,a],1,a)
	h = repmat(h[:,a],1,a)
	return accumulated_logw,h,z
else
	return accumulated_logw,h,z
end
end
@inline function local_lik(h,fc,XYZ,D,nz)
	D	= repeat(D,outer=(1,fc.nsim))
	XYZ[1]	= fc.yPY(send_effector(D)) # latent representation of x
	ϕ	= fc.Pϕ(vcat(h,XYZ[1:2]...))
	μσtmp	= fc.hPrior(h)
	μ 	= fc.hPriorμ(μσtmp)
	σs 	= fc.hPriorσ(μσtmp)
	L,z	= fc.Nz(ϕ)
	Lt	= normlpdf(z,μ,σs)
	L	= L .+ Lt
	XYZ[3]	= fc.zPZ(send_effector(z))
	θ	= fc.zPθ(vcat(XYZ[2:3]...))
	L 	= Tracker.collect([begin
			τ	= θ[4,k]*MainType(0.1)
			L[k]+boundτ(ddm(θ[1,k],θ[2,k],θ[3,k],τ,D[1,1],D[2,1]),τ,D[1,1])
		end for k in 1:fc.nsim]')

	L
end

function (fc::TCNChain)(rt,c,x)
	y = [rt c]

	x = hcat(x...)'

	Y = fc.CY(y)
	X = fc.CX(x)
	pz = fc.FZ(vcat(X,Y))
	μ = pz[1:fc.nz,:]
	σ = exp.(pz[fc.nz+1:end,:])
	z = μ .+ σ .* param(randn(fc.nz,size(pz,2)))
	L = normlpdf(z,0.0,1.0) .+ sum(pz[fc.nz+1:end,:] .+ MainType(log2π/2) .* 0.5 .+ 0.5,dims=1)
	θ = fc.FY(vcat(z,X))
	Lt 	= Tracker.collect([begin
			τ	= θ[4,k]*MainType(0.1)
			boundτ(ddm(θ[1,k],θ[2,k],θ[3,k],τ,rt[k],c[k]),τ,rt[k])
		end for k in 1:size(y,2)]')
	mean(L)+mean(Lt)
end

"""

	ddm(a::Real,v::Real,w::Real,τ::Real,rt::Real,c::Real)

	ddm logpdf

"""
function ddm(a,v,w,τ,rt,c)
	if c != zero(c)
		return SSM.DDM_lpdf(a,v,w,τ,rt,c)
	else
		return log(minimum((maximum((one(a)*floatmin(MainType),1 - SSM.DDM_cdf(a,v,w,τ,rt,c))),one(a))))
	end
end
const ddmv(x) = ddm(x...)
if isCu
	const ddmgrad = DiffResults.GradientResult(Float32.(zeros(6)))
	const cfg6 = ForwardDiff.GradientConfig(ddmv, Float32.([1.0;2.0;0.0;0.0;1.0;1.0]), ForwardDiff.Chunk{6}());
else
	const ddmgrad = DiffResults.GradientResult(zeros(6))
	const cfg6 = ForwardDiff.GradientConfig(ddmv, [1.0;2.0;0.0;0.0;1.0;1.0], ForwardDiff.Chunk{6}());
end
@grad function ddm(a,v,w,τ,rt,c)
	ForwardDiff.gradient!(ddmgrad,ddmv,data.([a,v,w,τ,rt,c]),cfg6)
	G = ∇->(ddmgrad.derivs[1][1]*∇,ddmgrad.derivs[1][2]*∇,ddmgrad.derivs[1][3]*∇,ddmgrad.derivs[1][4]*∇,∇*0,∇*0)
	return ddmgrad.value,G
end
ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float32,c::Float32) = Tracker.track(ddm,a,v,w,τ,rt,c)
ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float64,c::Float64) = Tracker.track(ddm,a,v,w,τ,rt,c)

function normlpdf(z::S , μ::T , σs::T) where {S,T}
	D = (z .- μ) ./ σs
	M = D.*D ./2 .- log.(σs) .- MainType(log2π/2)
	return .- sum(M ,dims=1)
end
