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
function (fc::FIVOChain)(RT,C,x;
	gradient_fetch_interval::Integer = -1, compute_intermediate_grad::Bool = false,opt_local=()->nothing,
	eval::Bool=false)
	if eval
		compute_intermediate_grad = false
		fc_out 			  = deepcopy(fc)
		fc_out.output.eval	  = true
		fc_out.output.θ	  	  = [[]]
		fc_out.output.log_w	  = [[]]
		fc_out.output.L	  	  = 0.0
	end
	if fc.GPU
		RT = Float32.(RT)
		C = Float32.(C)
		x = cu.(x)
	end
	if gradient_fetch_interval > 0
		init = rand(1:gradient_fetch_interval)
		seq_gradient_fetch_interval = init:gradient_fetch_interval:length(RT)
	else
		seq_gradient_fetch_interval = length(RT)+1:length(RT)+1
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
	accumulated_logw = param(-log(nsim) * ones(MainType,1,nsim))
	if fc.GPU
		accumulated_logw = gpu(accumulated_logw)
	end
	L = zero(MainType)

	local_lik = make_local_lik(fc,x,RT,C)
	for (t,(rt,c)) in enumerate(zip(RT,C))
		if t ∈ seq_gradient_fetch_interval
#			print("stack grad at ",t,"\n")
			# break dependency of the current log-lik on previous time steps
			if compute_intermediate_grad
				Tracker.back!(L)
				opt_local()
				L = data(L)
			end

			accumulated_logw 		= data(accumulated_logw)
			local_lik.Zt 			= param(data(local_lik.Zt))
			fc.G.state 			= param(data(fc.G.state))
		end

		log_alpha_t 			= local_lik(fc,t,rt,c)

		log_p_hat_t_summand 		= log_alpha_t .+ accumulated_logw
		log_p_hat_t 			= logsumexp_overflow(log_p_hat_t_summand)
		L 				= elinf(L + log_p_hat_t/ntrials)
		accumulated_logw 		= log_p_hat_t_summand .- log_p_hat_t
		if fc_out.output.eval
			push!(fc_out.output.log_w,data(log_p_hat_t_summand))
			push!(fc_out.output.log_w_unnormalized,data(accumulated_logw))
		end
		accumulated_logw 		= resample(accumulated_logw,fc.G,local_lik,fc.GPU)
	end
	if fc_out.output.eval
		fc_out.output.L = data(L)
	end
	if fc_out.output.eval
		return fc_out
	else
		return L
	end
end

mutable struct make_local_lik
	Xs
	Ys
	Zt
	function make_local_lik(fc,x,RT,C)
		GPU = fc.GPU
		nnodes = fc.nnodes
		nsim = fc.nsim

		# Fetch X's: latent representation of regressors
		Xs = fc.xPX(hcat(x...))
		# Fetch Y's: latent representation of data
		Ys = fc.yPY([RT C]')

		MainType = GPU ? Float32 : Float64
		Zt = param(zeros(MainType,nnodes,nsim))
		if GPU
			Zt = Zt |> gpu
		end
		fc.G.state = repeat(fc.G.state,outer=(1,nsim))
		new(Xs,Ys,Zt)
	end
end
function (local_lik::make_local_lik)(fc,t, rt ,c)
		Xt,Yt = local_lik.Xs[:,t], local_lik.Ys[:,t]
		Zt = local_lik.Zt
		GPU = fc.GPU
		h = fc.G.state

		ϕ	= fc.Pϕ((h,Xt,Yt))
		μσtmp	= fc.hPrior(h)
		μ 	= fc.hPriorμ(μσtmp)
		σs 	= fc.hPriorσ(μσtmp)
		nH,z	= fc.Nz(ϕ)
		Lt	= normlpdf(z,μ,σs)
		L	= Lt .- nH
		if GPU
			z = gpu(z)
		end
		Zt	= fc.zPZ(z)
		θ	= fc.zPθ((Yt,Zt))
		if GPU
			θ = θ |> cpu
		end
		Lt 	= Tracker.collect([begin
			τ	= GPU ? θ[4,k]*Float32(0.1) : θ[4,k] * 0.1
			boundτ(ddm(θ[1,k],θ[2,k],θ[3,k],τ,rt,c),τ,rt)
		end for k in 1:fc.nsim]')
		if GPU
			Lt = Lt |> gpu
		end

		fc.G((Xt,Yt,Zt)) # map previous regressors, data and latent variable to hidden state of GRU
		if fc_out.output.eval
			push!(fc_out.output.θ,data(θ))
		end

		L .+ Lt
end
function resample(accumulated_logw,G,local_lik,GPU)
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
	local_lik.Zt 			= repeat(local_lik.Zt[:,a],outer=(1,N))
	h 			= repeat(h[:,a],outer=(1,N))
	G.state 		= h # in place
	if GPU
		accumulated_logw = accumulated_logw |> gpu
	end
	return accumulated_logw
else
	return accumulated_logw
end
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
	M = D.*D ./2 .+ log.(σs) .+ TYPE(log2π/2)
	return .- sum(M ,dims=1)
end
