module RaeDm

macro mainDef(x)
    """
        Variables defined using @mainDef macro are overwritten in module if they exist in the Main environment.
        When this occurs, @mainDef warns the user about this.
    """
        xn = string(x.args[1])
        y = isdefined(Main,x.args[1]) ? :(Printf.@printf("Warning: Overwriting %s to %s\n",$xn,Main.$(x.args[1]));const $(x.args[1]) = Main.$(x.args[1])) : Expr(:const,x)
        esc(y)
end

@mainDef isCu = haskey(ENV,"CUDA_HOME") || haskey(ENV,"CUDA_PATH")
if isCu
    print("Using CUDA\n")
    using CuArrays
end
using Flux, ForwardDiff, DiffResults, StatsFuns, NNlib
using Flux.Tracker: TrackedReal, TrackedArray, track, @grad, data
using StatsFuns: softplus
import Flux: gpu, cpu

export FIVOChain, zero_grad!, gpu, cpu,optimize
include("miscelaneous.jl")

@mainDef DEBUG=false

include("ddm_distributions.jl")
include("types.jl")
include("truncgauss.jl")
include("ndt.jl")


include("model.jl")

end # module
