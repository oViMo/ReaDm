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

	XYZ = [param(zeros(MainType,nnodes,nsim)) for k in 1:3]
	if fc.GPU
		XYZ = gpu.(XYZ)
	end
	for (rt,c,k) in zip(RT,C,1:ntrials)
		Yt      = fc.xPX(x[k]) # latent representation of regressors
		h		= fc.G(vcat(XYZ...)) # map previous regressors, data and latent variable to hidden state of GRU
		XYZ[2] 	= repeat(Yt,outer=(1,nsim))
		D 		= [rt;c]
		log_alpha_t 			= local_lik(h,fc,XYZ,D,nz,fc.GPU)
		#@show  typeof(log_alpha_t)
		#@show  typeof(accumulated_logw)
		log_p_hat_t_summand 	= log_alpha_t .+ accumulated_logw
		log_p_hat_t 			= logsumexp_overflow(log_p_hat_t_summand)
		@show log_p_hat_t
		L 						= elinf(L + log_p_hat_t)
		accumulated_logw 		= log_p_hat_t_summand .- log_p_hat_t
		accumulated_logw,XYZ[3] = resample(accumulated_logw,fc.G,XYZ[3],fc.GPU)
	end
	L/ntrials
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
	a = findfirst(cumsum(exp.(acc_logw_detach[:]),dims=1) .> rand())
	accumulated_logw = -log(N)*param(ones(1,N))
	z = repeat(z[:,a],outer=(1,N))
	h = repeat(h[:,a],outer=(1,N))
	G.state = h # in place
	if GPU
		accumulated_logw = accumulated_logw |> gpu
	end
	return accumulated_logw,z
else
	return accumulated_logw,z
end
end

function optimize(F::FIVOChain,RT,C,X)
	opt = Adam(params(F), 0.001)

	for k in 1:1000
		ss = rand(1:length(RT))
		L = -F(RT[ss],C[ss],X[ss])
		Tracker.back!(L)
		opt()
		zero_grad!(F)
		if k % 10 == 0
			print("k = ",k,"\t L = ",L.data,"\r")
		end
	end
	print("\n")
	return F
end
@inline function local_lik(h,fc,XYZ,D,nz, GPU)
	D	= repeat(D,outer=(1,fc.nsim))
	if GPU
		D = gpu(D)
	end
	XYZ[1]	= fc.yPY(D) # latent representation of x
	ϕ	= fc.Pϕ(vcat(h,XYZ[1:2]...))
	μσtmp	= fc.hPrior(h)
	μ 	= fc.hPriorμ(μσtmp)
	σs 	= fc.hPriorσ(μσtmp)
	L,z	= fc.Nz(ϕ)
	#@show  typeof(L)
	#@show  typeof(z)
	Lt	= normlpdf(z,μ,σs)
	#@show  typeof(Lt)
	L	= L .+ Lt
	#@show  typeof(L)
	if GPU
		z = gpu(z)
	end
	XYZ[3]	= fc.zPZ(z)
	θ	= fc.zPθ(vcat(XYZ[2:3]...)) |> cpu
	#@show  typeof(θ)
	Lt 	= Tracker.collect([begin
			τ	= GPU ? θ[4,k]*Float32(0.1) : θ[4,k] * 0.1
			boundτ(ddm(θ[1,k],θ[2,k],θ[3,k],τ,D[1,1],D[2,1]),τ,D[1,1])
		end for k in 1:fc.nsim]')
	if GPU
		Lt = Lt |> gpu
	end

	##@show  typeof(L)

	L .+ Lt#

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
	if abs(c) == one(c)
		return SSM.DDM_lpdf(a,v,w,τ,rt,c)
	else
		MIN = one(a)*floatmin(typeof(one(rt)))
		cdf = SSM.DDM_cdf(a,v,w,τ,rt,1.0) + SSM.DDM_cdf(a,v,w,τ,rt,-1.0)
		return log(minimum((maximum((MIN ,1 - cdf)),one(a))))
	end
end
# const ddmv(x) = ddm(x...)
# if isCu
# 	const ddmgrad_float32 = DiffResults.GradientResult(Float32.(zeros(6)))
# 	const cfg6_float32 = ForwardDiff.GradientConfig(ddmv, Float32.([1.0;2.0;0.0;0.0;1.0;1.0]), ForwardDiff.Chunk{6}());
# end
# const ddmgrad_float64 = DiffResults.GradientResult(zeros(6))
# const cfg6_float64 = ForwardDiff.GradientConfig(ddmv, [1.0;2.0;0.0;0.0;1.0;1.0], ForwardDiff.Chunk{6}());
#
# @grad function ddm(a::Union{Flux.Tracker.TrackedReal{Float32},Float32},v,w,τ,rt,c)
# 	ForwardDiff.gradient!(ddmgrad_float32,ddmv,data.([a,v,w,τ,rt,c]),cfg6_float32)
# 	G = ∇->(ddmgrad_float32.derivs[1][1]*∇,ddmgrad_float32.derivs[1][2]*∇,ddmgrad_float32.derivs[1][3]*∇,ddmgrad_float32.derivs[1][4]*∇,∇*0,∇*0)
# 	return ddmgrad_float32.value,G
# end
# @grad function ddm(a::Union{Flux.Tracker.TrackedReal{Float64},Float64},v,w,τ,rt,c)
# 	ForwardDiff.gradient!(ddmgrad_float64,ddmv,data.([a,v,w,τ,rt,c]),cfg6_float64)
# 	G = ∇->(ddmgrad_float64.derivs[1][1]*∇,ddmgrad_float64.derivs[1][2]*∇,ddmgrad_float64.derivs[1][3]*∇,ddmgrad_float64.derivs[1][4]*∇,∇*0,∇*0)
# 	return ddmgrad_float64.value,G
# end
# ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float32,c::Float32) = Tracker.track(ddm,a,v,w,τ,rt,c)
# ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float64,c::Float64) = Tracker.track(ddm,a,v,w,τ,rt,c)

function normlpdf(z::S , μ::T , σs::T) where {S,T}
	D = (z .- μ) ./ σs
	TYPE = typeof(D[1])
	M = D.*D ./2 .- log.(σs) .- TYPE(log2π/2)
	return .- sum(M ,dims=1)
end
