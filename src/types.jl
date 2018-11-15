###### Types #######
#==================#
include("NF.jl")
import Flux: params, mapleaves, children, mapchildren, @treelike,glorot_uniform,gate,reset!, Recur, hidden, Chain
import Flux.Tracker: zero_grad!
#	mutable struct Particles{T}
#		X::T
#		Z::T
#		Y::T
#	end

mutable struct FIVOout
	eval::Bool
	θ::Array{AbstractArray,1}
	log_w::Array{AbstractArray,1}
	log_w_unnormalized::Array{AbstractArray,1}
	L::Real
	function FIVOout()
		new(false,[[]],[[]],[[]],0.0)
	end
end
struct FIVOChain{N,R}
	Nz::N
	G::R
	yPY::Flux.Chain
	zPZ::Flux.Chain
	xPX::Flux.Chain
	Pϕ::Flux.Chain
	zPθ::Flux.Chain
	hPrior::Dense
	hPriorμ::Dense
	hPriorσ::Dense
	nsim::Int64
	nnodes::Int64
	nlayers::Int64
	nx::Int64
	ny::Int64
	nz::Int64
	output::FIVOout
	GPU::Bool

	function FIVOChain(;nx::Int64=2,ny::Int64=2,nz::Int64=10,nlayers::Int64=0,nnodes::Int64=50,nsim::Int64=4,afun=elu)
		Nz	= NF(nz,nlayers)
		G 	= GRU_mult((nnodes,nnodes,nnodes),nnodes)
		yPY 	= Chain(Dense(ny,nnodes,afun),Dense(nnodes,nnodes,afun)) # data
		zPZ 	= Chain(Dense(nz,nnodes,afun),Dense(nnodes,nnodes,afun)) # latent input
		xPX	= Chain(Dense(nx,nnodes,afun),Dense(nnodes,nnodes,afun))  # regressor

		Pϕ	= Chain(Dense_mult((nnodes,nnodes,nnodes),nnodes,afun),
				Dense(nnodes,Nz.np))

		zPθ	= Chain(Dense_mult((nnodes,nnodes),nnodes,afun),
				Dense(nnodes,nnodes,afun),Dense(nnodes,4))

		hPprior = Dense(nnodes,nnodes,afun)
		hPpriorμ = Dense(nnodes,nz)
		hPpriorσ = Dense(nnodes,nz,NNlib.softplus)

		# S = Particles(param(zeros(nnodes,nsim)),param(zeros(nnodes,nsim)),param(zeros(nnodes,nsim)))

		new{NF,Flux.Recur}(Nz,G,yPY,zPZ,xPX,Pϕ,zPθ,hPprior,hPpriorμ,hPpriorσ,
						   nsim,nnodes,nlayers,nx,ny,nz,
						   FIVOout(),
						   false)
	end
	function FIVOChain(Nz,G,yPY,zPZ,xPX,Pϕ,zPθ,hPprior,hPpriorμ,hPpriorσ,nsim,nnodes,nlayers,nx,ny,nz,
			   output=FIVOout(),
			   GPU=false)
		F = new{NF,Flux.Recur}(Nz,G,yPY,zPZ,xPX,Pϕ,zPθ,hPprior,hPpriorμ,hPpriorσ,nsim,nnodes,nlayers,nx,ny,nz,output,GPU)
	end
end
gpu(x::FIVOChain) = FIVOChain(map(f->gpu(getfield(x,f)),fieldnames(FIVOChain)[1:end-1])...,true)

function squeeze(x)
       s = vcat(size(x)...);s2 = Tuple(s[s .!= 1])
       reshape(x,s2)
end
function reshape_out_tcn(x)
	squeeze(permutedims(x,(3,2,1,4)))
end
function reshape_in_tcn(x::AbstractArray{<:Real,2})
	permutedims(reshape(x,(size(x)...,1,1)),(3,1,2,4))
end
@treelike FIVOChain
struct TCNChain
	CX::Flux.Chain
	CY::Flux.Chain
	FZ::Flux.Chain
	FY::Flux.Chain
	num_zero
	nsim::Int64
	nnodes::Int64
	nlayers::Int64
	nx::Int64
	ny::Int64
	nz::Int64
	k::Int64
	pad::Int64
	dilation::Int64


	function TCNChain(;nlayers::Int64=1,nsim::Int64=1,nx::Int64=2,ny::Int64=2,
			  k::Int64=3,nnodes::Int64=20,pad::Int64=0,dilation::Int64=2,nz::Int64=4,stride::Int64=1)
		CY = Chain(reshape_in_tcn,
			   Conv((1,k),ny=>nnodes,relu,pad=pad,stride=stride),
			  Conv((1,k),nnodes=>nnodes,relu,dilation=dilation,pad=pad,stride=stride),
			  Conv((1,k),nnodes=>nnodes,relu,dilation=dilation^2,pad=pad,stride=stride),
			  reshape_out_tcn)
		CX = Chain(reshape_in_tcn,
			   Conv((1,k),nx=>nnodes,relu,pad=pad,stride=stride),
			  Conv((1,k),nnodes=>nnodes,relu,dilation=dilation,pad=pad,stride=stride),
			  Conv((1,k),nnodes=>nnodes,relu,dilation=dilation^2,pad=pad,stride=stride),
			  reshape_out_tcn)
		FZ = Chain(Dense(2*nnodes,nnodes,relu),Dense(nnodes,2nz))
		FY = Chain(Dense(nz+nnodes,nnodes,relu),Dense(nnodes,nnodes),Dense(nnodes,4))
		num_zero(l,lev=0) = begin
			t = 0
			d = 2
			for i in lev:-1:0
				l=l + k*d^lev - l - stride + l*stride
			end
			l
		end
		new(CY,CX,FZ,FY,
		    num_zero,
		    nsim,nnodes,nlayers,nx,ny,nz,k,pad,dilation)
	end
end
@treelike TCNChain
function zero_grad!(C::FIVOChain)
	map(x->x.grad .= zero(x.grad[1]),params(C))
	nothing
end
function reset!(C::FIVOChain)
#	C.S.X = zero.(C.S.X)
#	C.S.Y = zero.(C.S.Y)
#	C.S.Z = zero.(C.S.Z)

	#C.G.cell.h = zero.(C.G.cell.h)
	#C.G.state = zero.(C.G.state)
	#C.G.init = zero.(C.G.init)
	reset!(C.G)
	nothing
end







"""
Dense_mult(in::NTuple{N,Integer}, out::Integer, σ = identity) where N::Integer
Creates a traditional `Dense_mult` layer with parameters `W` and `b` for multiple inputs (avoids repeat and cat).
    y = σ.(W * x .+ b)
The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.
```julia
julia> d = Dense_mult((5,8), 2)
Dense_mult((5,8), 2)
julia> d(rand(5),randn(8))
Tracked 2-element Array{Float64,1}:
  0.00257447
  -0.00449443
```
"""
struct Dense_mult{F,S,T}
  W::S
  b::T
  σ::F
end

#params(x::Dense_mult) = [x.W... , x.b]
Dense_mult(W, b) = Dense_mult(W, b, identity)

function Dense_mult(in::NTuple{N,Integer}, out::Integer, σ = identity;
		    initW = glorot_uniform, initb = zeros) where N
	return Dense_mult(map(in->param(initW(out, in)),in), param(initb(out)), σ)
end

@treelike Dense_mult

function (a::Dense_mult)(x)
  W, b, σ = a.W, a.b, a.σ
  P = W[1]*x[1]
  for k in 2:length(W)
	  P = P .+ W[k]*x[k]
  end
  σ.(P .+ b)
end

function Base.show(io::IO, l::Dense_mult)
  print(io, "Dense_mult(", size.(l.W, 2), ", ", size(l.W[1], 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end








# GRU

mutable struct GRUCell_mult{A,V,K}
  Wi::NTuple{K, A}
  Wh::A
  b::V
  h::V
end

function GRUCell_mult(in::NTuple{N,Integer}, out; init = glorot_uniform) where N
    GRUCell_mult( map(in->param(init(out*3, in)),in), param(init(out*3, out)),
          param(zeros(out*3)), param(init(out)))
end

function (m::GRUCell_mult)(h, x)
  b, o = m.b, size(h, 1)
  gh = m.Wh*h
  gx = m.Wi[1]*x[1]
  for k in 2:length(x)
	  gx = gx .+ m.Wi[k]*x[k]
  end
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

hidden(m::GRUCell_mult) = m.h

@treelike GRUCell_mult

Base.show(io::IO, l::GRUCell_mult) =
	print(io, "GRUCell_mult(", size.(l.Wi, 2), ", ", size(l.Wi[1], 1)÷3, ")")

"""
	GRU_mult(in::NTuple{N,Integer}, out::Integer)
Gated Recurrent Unit layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.
See [this article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
GRU_mult(a...; ka...) = Recur(GRUCell_mult(a...; ka...))




# local_lik carries the likelihood of each particle
mutable struct make_local_lik
	Xs::TrackedArray
	Ys::TrackedArray
	Zt::TrackedArray
	log_alpha_t::TrackedArray
	log_p_hat_t::TrackedReal
	L::TrackedReal
	accumulated_logw::TrackedArray
	
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
		
		MainType = fc.GPU ? Float32 : Float64
		accumulated_logw = param(-log(nsim) * ones(MainType,1,nsim))
		fc.GPU && (accumulated_logw = gpu(accumulated_logw))
		L = param(zero(MainType))
		new(Xs,Ys,Zt,
		    param(zeros(MainType,1,fc.nsim)),param(zero(MainType)),L,accumulated_logw)
	end
end
