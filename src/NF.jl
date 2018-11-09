using LinearAlgebra: dot
using NNlib
dotdim(x,y,d) = sum(x .* y,dims=d)
#dotdim(x,y,d) = reduce(+,x .* y,dims=d)
#============= NF =============#
#==============================#
struct NF
	nz::Int64
	nlayers::Int64
	np::Int64
	idμ
	idσ
	idvw
	idvu
	idb
	GPU::Bool
	function NF(nz::Int64,nlayers::Int64)#,np::Int64)
		id = 0
		idμ = id.+(1:nz)
		id      += nz
		idσ = id.+(1:nz)
		id      += nz

		idvw = [id.+(1:nz)]
		id      += nz
		idvu = [id.+(1:nz)]
		id      += nz
		idb = [id+1]
		id      += 1
		if nlayers>1
			for k in 2:nlayers
				push!(idvw,id.+(1:nz))
				id      += nz
				push!(idvu,id.+(1:nz))
				id      += nz
				push!(idb,id+1)
				id	+= 1
			end
		end
		np = 2nz+2*nlayers*nz+nlayers
		new(nz,nlayers,np,idμ,idσ,idvw,idvu,idb,false)
	end
	function NF(nz,nlayers,np,idμ,idσ,idvw,idvu,idb,GPU)
		new(nz,nlayers,np,idμ,idσ,idvw,idvu,idb,GPU)
	end
end
gpu(x::NF) =  NF(map(f->gpu(getfield(x,f)),fieldnames(NF)[1:end-1])...,true)

function (a::NF)(x)
	NFfun(x,a,a.nz,a.nlayers,a.GPU)
end
function NFfun(x,a,nz,nlayers,GPU=false)
	if GPU
		MainType = Float32
		send_effector = gpu
	else
		MainType = Float64
		send_effector(x) = x
	end
	if GPU
		# Float32
		 Zero = 0.0f0
		 OneHalf = 0.5f0
		 log2πHalf = 0.9189385f0
		 One = 1.0f0
	else
		# Float64
		 Zero = 0.0
		 OneHalf = 0.5
		 log2πHalf = 0.9189385332046727
		 One = 1.0
	end

	μ 	= getindex(x,a.idμ,:)
	σ 	= NNlib.softplus.((getindex(x,a.idσ,:)))

	EntropyComp = zeros(MainType,(1,size(x,2)))
	r = send_effector(randn(MainType,(nz,size(x,2))))
	z = μ .+ σ .* r
	M = .- (OneHalf .* log.(σ) .+ log2πHalf)
	EntropyComp = sum(M, dims=1)
	for k in 1:nlayers
		vw 	= getindex(x,a.idvw[k],:)
		vu 	= getindex(x,a.idvu[k],:)
		b	= getindex(x,a.idb[k],:)'
		ett,z 	= subNFfun(vw,vu,b,z,One)
		EntropyComp = EntropyComp .- ett
	end
	EntropyComp,z
end
function subNFfun(vw,vu,b,z,One)
	dotvwvu	= dotdim(vw,vu,1)
	vu 	= vu .+ ((NNlib.softplus.(dotvwvu) .- One .- dotvwvu) .* vw ./ dotdim(vw,vw,1))
	Hval 	= tanh.(dotdim(vw , z,1) .+ b)
	δHval 	= -Hval.*Hval .+ One
	EntropyComp = log.(abs.(One .+ dotdim(vu,vw,1) .* δHval))
	z	= z .+ vu .* Hval
	EntropyComp,z
end
##============= NF ndt =============#
##==================================#
#
#struct NFτ
#	nlayers::Int64
#	nz::Int64
#	np::Int64
#end
#function NFτ(nlayers::Int64,nz::Int64)
#	NFτ(nlayers,nz,2nz+nlayers*nz+nlayers*(nz-1)+nlayers)
#end
#function (a::NFτ)(x,lim::Real)
#	NFτfun(x,lim,a.nz,a.nlayers)
#end
#function NFτfun(x,lim,nz,nlayers)
#	id 	= 0
#	μ 	= getindex(x,id.+(1:nz))
#	id	+= nz
#	σ 	= softplus.(getindex(x,id.+(1:nz)))
#	id	+= nz
#
#	EntropyComp = 0.0
#	r = randn(length(σ)-1)
#	z = μ[1:end-1] + σ[1:end-1] .* r
#	z = vcat(z,rtnuni(μ[end],σ[end],lim))
#	lmax = length(μ)
#	for k in 1:lmax
#		EntropyComp -= k==lmax ? -entropyTMN(μ[k],σ[k],lim) : .5 + log(σ[k]) + 0.9189385332046727
#	end
#	for k in 1:nlayers
#		vw 	= getindex(x,id.+(1:nz))
#		id+=nz
#		vu 	= getindex(x,id.+(1:nz-1))
#		id+=nz-1
#		b	= getindex(x,id+1)
#		id+=1
#		ett,z 	= subNFτfun(vw,vu,b,z,nz)
#		EntropyComp -= ett
#	end
#	EntropyComp,z
#end
#
#function subNFτfun(vw,vu,b,z,nz)
#	vw2	= vw[1:end-1]
#	dotvwvu	= dot(vw2,vu)
#	vu 	= vu + ((softplus(dotvwvu) - 1.0 - dotvwvu) .* vw2 ./ dot(vw2,vw2))
#	vu2	= vcat(vu,0.0)
#	Hval 	= tanh(dot(vw , z) + b)
#	δHval 	= -Hval*Hval + 1.0
#	EntropyComp = log(abs(1.0 + dot(vu2,vw)*δHval))
#	z	= z + vu2 .* Hval
#	EntropyComp,z
#end
