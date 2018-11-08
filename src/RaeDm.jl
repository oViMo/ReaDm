module RaeDm

const isCu = haskey(ENV,"CUDA_HOME")

export FIVOChain, Tracker, Flux

using Flux, ForwardDiff, DiffResults, StatsFuns, NNlib
using Flux.Tracker: TrackedReal, TrackedArray, track, @grad, data
using StatsFuns: softplus

include("ddm_distributions.jl")
include("types.jl")
include("truncgauss.jl")
include("ndt.jl")
include("miscelaneous.jl")


include("model.jl")

end # module
git add * ; git commit -m "init" ; git push --set-upstream origin master
