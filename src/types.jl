###### Types #######
#==================#
include("NF.jl")
import Flux: params, mapleaves, children, mapchildren, @treelike,glorot_uniform,gate,reset!
import Flux.Tracker: zero_grad!
mutable struct Particles{T}
	X::T
	Z::T
	Y::T
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
#	S::Particles
	nsim::Int64
	nnodes::Int64
	nlayers::Int64
	nx::Int64
	ny::Int64
	nz::Int64

	function FIVOChain(;nx::Int64=2,ny::Int64=1,nz::Int64=10,nlayers::Int64=4,nnodes::Int64=50,nsim::Int64=4)
		Nz	= NF(nz,nlayers)
		G 	= GRU(3nnodes,nnodes)
		yPY 	= Chain(Dense(ny,nnodes,afun),Dense(nnodes,nnodes,afun)) # data
		zPZ 	= Chain(Dense(nz,nnodes,afun),Dense(nnodes,nnodes,afun)) # latent input
		xPX	= Chain(Dense(nx,nnodes,afun),Dense(nnodes,nnodes,afun)) # regressor

		Pϕ	= Chain(Dense(3nnodes,nnodes,afun),
				Dense(nnodes,Nz.np))

		zPθ	= Chain(Dense(2nnodes,nnodes,afun),
				Dense(nnodes,nnodes,afun),Dense(nnodes,4))

		hPprior = Dense(nnodes,nnodes,afun)
		hPpriorμ = Dense(nnodes,nz)
		hPpriorσ = Dense(nnodes,nz,NNlib.softplus)

		S = Particles(param(zeros(nnodes,nsim)),param(zeros(nnodes,nsim)),param(zeros(nnodes,nsim)))
		new{NF,Flux.Recur}(Nz,G,yPY,zPZ,xPX,Pϕ,zPθ,hPprior,hPpriorμ,hPpriorσ,
				   #S,
				   nsim,nnodes,nlayers,nx,ny,nz)
	end
	function FIVOChain(Nz,G,yPY,zPZ,xPX,Pϕ,zPθ,hPprior,hPpriorμ,hPpriorσ,nsim,nnodes,nlayers,nx,ny,nz)
		new{NF,Flux.Recur}(Nz,G,yPY,zPZ,xPX,Pϕ,zPθ,hPprior,hPpriorμ,hPpriorσ,nsim,nnodes,nlayers,nx,ny,nz)
	end
end
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
