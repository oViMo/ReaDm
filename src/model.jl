#@mainDef afun = elu
const afun = elu

## Exec ##
#========#

function (fc::FIVOChain)(RT::Array{Float64,1},
			 C,
			 x,
			 args... ; kwargs...)
	@warn "Data should be cast to Float32 for efficiency"
	return fc(Float32.(RT), Float32.(C), map(_x->Float32.(_x), x), args... ; kwargs...)
end
function (fc::FIVOChain)(
			 RT::Array{Float32,1},
			 C::Array{Float32,1},
			 x::Array{Array{Float32,1},1};
			gradient_fetch_interval::Integer = -1, 
			compute_intermediate_grad::Bool = false, 
			opt_local=()->nothing, 
			single_update::Bool=false,
			eval::Bool=false, isCuda = isCuda[])
	if eval
		compute_intermediate_grad = false
		zero_grad!(fc)
		reset!(fc.G)
		fc 		  = FIVOChain(map(f->getfield(fc,f),fieldnames(FIVOChain)[1:end-2])...,FIVOout(),fc.GPU) # we leave fc untouched
		fc.output.eval	  = true
		fc.output.θ	  = [[]]
		fc.output.log_w	  = [[]]
		fc.output.L	  = 0.0f0
	end
	if isCuda
		x = cu.(x)
	end

	if gradient_fetch_interval > 0
		init = rand(1:gradient_fetch_interval)
		likelihood_stack_grad_list = vcat(collect(init:gradient_fetch_interval:length(RT)),length(RT))
		if single_update
			opt_step = [rand(likelihood_stack_grad_list)]
		else
			opt_step = likelihood_stack_grad_list
		end
	else
		gradient_fetch_interval = length(RT)
		likelihood_stack_grad_list = [length(RT)+1]
	end
	nsim	= fc.nsim
	nnodes  = fc.nnodes
	nlayers = fc.nlayers
	nx	= fc.nx
	ny	= fc.ny
	nz	= fc.nz

	reset!(fc.G)


	local_lik = make_local_lik(fc,x,RT,C)
	trials_since_last = 0

	for (t,(rt,c)) in enumerate(zip(RT,C))
		trials_since_last += 1
		if t ∈ likelihood_stack_grad_list
#			print("stack grad at ",t,"\n")
			# break dependency of the current log-lik on previous time steps
			if compute_intermediate_grad && t ∈ opt_step
				Lp = -local_lik.L/gradient_fetch_interval
				Tracker.back!(Lp)
				opt_local()
				local_lik.L = param(data(local_lik.L))
				trials_since_last = 0
			elseif compute_intermediate_grad
				local_lik.L = param(data(local_lik.L)) # reset L if we don't care about L so far
			end
			local_lik.accumulated_logw 		= param(data(local_lik.accumulated_logw))
			local_lik.Zt 				= param(data(local_lik.Zt))
			fc.G.state 				= param(data(fc.G.state))
		end
		local_lik(fc,t,rt,c)
	end
	if fc.output.eval
		fc.output.L = data(local_lik.L)
	end
	if fc.output.eval
		return fc
	else
		return -local_lik.L/length(RT)
	end
end

function compute_loss(local_lik,fc)
log_alpha_t = local_lik.log_alpha_t
log_p_hat_t = local_lik.log_p_hat_t
accumulated_logw = local_lik.accumulated_logw
L = local_lik.L

log_p_hat_t_summand 		= log_alpha_t .+ accumulated_logw
log_p_hat_t 			= logsumexp_overflow(log_p_hat_t_summand)
L 				= elinf(L + log_p_hat_t)
accumulated_logw 		= log_p_hat_t_summand .- log_p_hat_t
if fc.output.eval
	push!(fc.output.log_w_unnormalized,data(log_p_hat_t_summand))
	push!(fc.output.log_w,data(accumulated_logw))
end
accumulated_logw 		= resample(accumulated_logw,fc.G,local_lik,fc.GPU)

local_lik.log_alpha_t = log_alpha_t
local_lik.log_p_hat_t = log_p_hat_t
local_lik.accumulated_logw = accumulated_logw
local_lik.L = L
nothing
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
			τ	= θ[4,k]*0.1f0
			boundτ(ddm(θ[1,k],θ[2,k],θ[3,k],τ,rt,c),τ,rt)
		end for k in 1:fc.nsim]')
		if GPU
			Lt = Lt |> gpu
		end

		fc.G((Xt,Yt,Zt)) # map previous regressors, data and latent variable to hidden state of GRU
		if fc.output.eval
			push!(fc.output.θ,data(θ))
		end

		local_lik.log_alpha_t = L .+ Lt
		compute_loss(local_lik,fc)
		nothing
end
function resample(accumulated_logw,G,local_lik,GPU)
acc_logw_detach = copy(accumulated_logw.data)
N = length(acc_logw_detach)
if -logsumexp_overflow(2 * acc_logw_detach) < log(N)-log(2)
	h = G.state
	if DEBUG[]
		print("\nResampling\n")
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
		return log(boundcdf(1.0f0 - (SSM.DDM_cdf(a,v,w,τ,rt,1.0f0) + SSM.DDM_cdf(a,v,w,τ,rt,-1.0f0))))
	end
end
const ddmv(x) = ddm(x...)

const ddmgrad_float32 = DiffResults.GradientResult(zeros(Float32, 6))
const cfg6_float32 = ForwardDiff.GradientConfig(ddmv, Float32[1.0;2.0;0.0;0.0;1.0;1.0], ForwardDiff.Chunk{6}());

@grad function ddm(a::Union{Flux.Tracker.TrackedReal{Float32},Float32},v,w,τ,rt,c)
	ForwardDiff.gradient!(ddmgrad_float32,ddmv,data.([a,v,w,τ,rt,c]),cfg6_float32)
	G = ∇->(ddmgrad_float32.derivs[1][1]*∇, 
		ddmgrad_float32.derivs[1][2]*∇, 
		ddmgrad_float32.derivs[1][3]*∇, 
		ddmgrad_float32.derivs[1][4]*∇, 
		∇*0, ∇*0)
	return ddmgrad_float32.value, G
end

ddm(a::TrackedReal,v::TrackedReal,w::TrackedReal,τ::TrackedReal,rt::Float32,c::Float32) = Tracker.track(ddm,a,v,w,τ,rt,c)

function normlpdf(z::S , μ::T , σs::T) where {S,T}
	D = (z .- μ) ./ σs
	M = D.*D ./2 .+ log.(σs) .+ Float32(log2π/2)
	return .- sum(M ,dims=1)
end
