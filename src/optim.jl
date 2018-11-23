
mutable struct optimize
	fc_out::Array
	optimizer
	mL::Array
	L::Array
	function optimize(optimizer)
		return new([],optimizer,[],[])
	end
end
function (OPT::optimize)(F::FIVOChain,RT,C,X;continuous_opt::Bool=true,niter=10000,report_interval=100,compute_intermediate_grad=false,kwargs...)
	opt = OPT.optimizer
	if continuous_opt
		opt_local = ()->begin
				opt()
				zero_grad!(F)
				end
	else
		opt_local = ()->nothing
	end

	for t in 1:niter
		ss = rand(1:length(RT))
		L = F(RT[ss],C[ss],X[ss];opt_local=opt_local,compute_intermediate_grad=compute_intermediate_grad,kwargs...)
		push!(OPT.mL,data(L))
		if !continuous_opt
			!compute_intermediate_grad && Tracker.back!(L)
			opt()
			zero_grad!(F)
		end
		if t % report_interval == 0 || t == 1
			OPT.fc_out = []
			for ss in 1:length(RT)
				push!(OPT.fc_out, F(RT[ss],C[ss],X[ss],eval=true))
			end
			mL = mean(map(x->x.output.L,OPT.fc_out))
			push!(OPT.mL,data(mL))
			print("t = ",t,"\t L = ",mL,"\n")
		end

	end
	print("\n")
	return OPT
end
