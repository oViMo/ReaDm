
mutable struct optimize
	fc_out
	optimizer
	function optimize(optimizer)
		return new([],optimizer)
	end
end
function (OPT::optimize)(F::FIVOChain,RT,C,X;gradient_fetch_interval::Integer=200,continuous_opt::Bool=true)
	opt = OPT.optimizer
	if continuous_opt
		opt_local = ()->begin
				opt()
				zero_grad!(F)
				end
	else
		opt_local = ()->nothing
	end

	OPT.fc_out = []
	for t in 1:1000
		ss = rand(1:length(RT))
		L = -F(RT[ss],C[ss],X[ss],gradient_fetch_interval=gradient_fetch_interval,opt_local=opt_local)
		Tracker.back!(L)
		opt()
		zero_grad!(F)
		if t % 10 == 0 || t == 1
			for ss in 1:length(RT)
				OPT.fc_out[ss] = F(RT[ss],C[ss],X[ss],eval=true)
			end
			print("t = ",t,"\t L = ",mean(map(x->x.L,OPT.fc_out)),"\n")
		end

	end
	print("\n")
	return OPT
end
