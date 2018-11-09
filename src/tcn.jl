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
