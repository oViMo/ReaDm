#@mainDef afun = elu
const afun = elu

## Cuda ##
#========#
# if isCu
# 	#import Base: exp,log1p
# 	#import StatsFuns:log1pexp,softplus
# 	const MainType = Float32
# 	function send_effector(x::AbstractArray)
# 		isempty(x) ? x : (x |> gpu)
# 	end
# 	function send_effector(x)
# 		x |> gpu
# 	end
# 	function send_effector(x::T) where {T<:Union{FIVOChain,Flux.Chain,Flux.Dense}}
# 		mapleaves(cu,x)
# 	end
# 	function send_effector(x::T) where {T<:Union{AbstractArray{<:AbstractArray},Tuple,NamedTuple}}
# 		map(send_effector,x)
# 	end
# else
# 	print("running on CPU\n")
# 	const MainType = Float64
# 	function send_effector(x)
# 		x
# 	end
# end

## Exec ##
#========#
function (fc::FIVOChain)(RT,C,x)
	if fc.GPU
		RT = Float32.(RT)
		C = Float32.(C)
		x = cu.(x)
	end
	nsim	= fc.nsim
	nnodes  = fc.nnodes
	nlayers = fc.nlayers
	nx	= fc.nx
	ny	= fc.ny
	nz	= fc.nz

	reset!(fc.G)

	ntrials = length(RT)
	MainType = typeof(RT[1])
	accumulated_logw = param(zeros(MainType,1,nsim))
	if fc.GPU
		accumulated_logw = gpu(accumulated_logw)
	end
	L = zero(MainType)

	Xt = param(zeros(MainType,nnodes))
	Yt = param(zeros(MainType,nnodes))
	Zt = param(zeros(MainType,nnodes,nsim))
	if fc.GPU
		XYZ = gpu.(XYZ)
	end

	# Fetch X's: latent representation of regressors
	Xs = fc.xPX(hcat(x...))
	# Fetch Y's: latent representation of data
	Ys = fc.yPY([RT C]')

	for (t,(rt,c)) in enumerate(zip(RT,C))
		h		= fc.G((Xt,Yt,Zt)) # map previous regressors, data and latent variable to hidden state of GRU
		
		Xt      	= Xs[:,t]
		Yt		= Ys[:,t]
		log_alpha_t,Zt 		= local_lik(h,fc,(Xt,Yt,Zt),nz,fc.GPU,rt,c)
		log_p_hat_t_summand 	= log_alpha_t .+ accumulated_logw
		log_p_hat_t 		= logsumexp_overflow(log_p_hat_t_summand)
		L 			= elinf(L + log_p_hat_t)
		accumulated_logw 	= log_p_hat_t_summand .- log_p_hat_t
		accumulated_logw,Zt 	= resample(accumulated_logw,fc.G,Zt,fc.GPU)
	end
	L/ntrials
end
@inline function local_lik(h,fc,XYZ,nz, GPU, rt ,c)
	ϕ	= fc.Pϕ((h,XYZ[1:2]...))
	μσtmp	= fc.hPrior(h)
	μ 	= fc.hPriorμ(μσtmp)
	σs 	= fc.hPriorσ(μσtmp)
	L,z	= fc.Nz(ϕ)
	Lt	= normlpdf(z,μ,σs)
	L	= L .+ Lt
	if GPU
		z = gpu(z)
	end
	Zt	= fc.zPZ(z)
	θ	= fc.zPθ((XYZ[2],Zt)) 
	if GPU
		θ = θ |> cpu
	end
	Lt 	= Tracker.collect([begin
			τ	= GPU ? θ[4,t]*Float32(0.1) : θ[4,t] * 0.1
			boundτ(ddm(θ[1,t],θ[2,t],θ[3,t],τ,rt,c),τ,rt)
		end for t in 1:fc.nsim]')
	if GPU
		Lt = Lt |> gpu
	end

	L .+ Lt, Zt

end
function resample(accumulated_logw,G,z,GPU)
acc_logw_detach = copy(accumulated_logw.data)
N = length(acc_logw_detach)
if -logsumexp_overflow(2 * acc_logw_detach) < log(N)-log(2)
	h = G.state
	if DEBUG
		print("\nResampling\n")
		##@show  acc_logw_detach
	end
	a 			= findfirst(cumsum(exp.(acc_logw_detach[:]),dims=1) .> rand())
	accumulated_logw 	= -log(N)*param(ones(1,N))
	z 			= repeat(z[:,a],outer=(1,N))
	h 			= repeat(h[:,a],outer=(1,N))
	G.state 		= h # in place
	if GPU
		accumulated_logw = accumulated_logw |> gpu
	end
	return accumulated_logw,z
else
	return accumulated_logw,z
end
end

function optimize(F::FIVOChain,RT,C,X)
	opt = Flux.ADAM(params(F), 0.001)

	for t in 1:1000
		ss = rand(1:length(RT))
		L = -F(RT[ss],C[ss],X[ss])
		Tracker.back!(L)
		opt()
		zero_grad!(F)
		if t % 10 == 0
			print("t = ",t,"\t L = ",L.data,"\r")
		end
	end
	print("\n")
	return F
end


"""

	ddm(a::Real,v::Real,w::Real,τ::Real,rt::Real,c::Real)

	ddm logpdf

"""
function ddm(a,v,w,τ,rt,c)
	if abs(c) == one(c)
		return SSM.DDM_lpdf(a,v,w,τ,rt,c)
	else
		return log(boundcdf(1.0 - (SSM.DDM_cdf(a,v,w,τ,rt,1.0) + SSM.DDM_cdf(a,v,w,τ,rt,-1.0))))
	end
end
const ddmv(x) = ddm(x...)
if isCu
	const ddmgrad_float32 = DiffResults.GradientResult(Float32.(zeros(6)))
	const cfg6_float32 = ForwardDiff.GradientConfig(ddmv, Float32.([1.0;2.0;0.0;0.0;1.0;1.0]), ForwardDiff.Chunk{6}());
end
const ddmgrad_float64 = DiffResults.GradientResult(zeros(6))
const cfg6_float64 = ForwardDiff.GradientConfig(ddmv, [1.0;2.0;0.0;0.0;1.0;1.0], ForwardDiff.Chunk{6}());

@grad function ddm(a::Union{Flux.Tracker.TrackedReal{Float32},Float32},v,w,τ,rt,c)
	ForwardDiff.gradient!(ddmgrad_float32,ddmv,data.([a,v,w,τ,rt,c]),cfg6_float32)
	G = ∇->(ddmgrad_float32.derivs[1][1]*∇,ddmgrad_float32.derivs[1][2]*∇,ddmgrad_float32.derivs[1][3]*∇,ddmgrad_float32.derivs[1][4]*∇,∇*0,∇*0)
	return ddmgrad_float32.value,G
end
@grad function ddm(a::Union{Flux.Tracker.TrackedReal{Float64},Float64},v,w,τ,rt,c)
	ForwardDiff.gradient!(ddmgrad_float64,ddmv,data.([a,v,w,τ,rt,c]),cfg6_float64)
	G = ∇->(ddmgrad_float64.derivs[1][1]*∇,ddmgrad_float64.derivs[1][2]*∇,ddmgrad_float64.derivs[1][3]*∇,ddmgrad_float64.derivs[1][4]*∇,∇*0,∇*0)
	return ddmgrad_float64.value,G
end
ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float32,c::Float32) = Tracker.track(ddm,a,v,w,τ,rt,c)
ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float64,c::Float64) = Tracker.track(ddm,a,v,w,τ,rt,c)

function normlpdf(z::S , μ::T , σs::T) where {S,T}
	D = (z .- μ) ./ σs
	TYPE = typeof(D[1])
	M = D.*D ./2 .- log.(σs) .- TYPE(log2π/2)
	return .- sum(M ,dims=1)
end
