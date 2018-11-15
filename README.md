# ReaDm
Recurrent auto-encoding Diffusion model

This is a Julia ([Flux](https://github.com/FluxML/Flux.jl))-based implementation of the [Recurrent Auto-Encoding DDM](https://www.biorxiv.org/content/early/2018/05/13/220517). 

Package is under development and without any guarantee of any kind.

Long term plan is to develop a FIVO package that would be used as a dipendency for RaeDm, which would just tweak it to fit the DDM problem.

Activate the package
--------------------
RaeDm is currently not a registered Julia package. 
To use it, copy the git repo by executing
`<git clone https://github.com/vmoens/RaeDm.git>`
`<git checkout dev>`

Next, the package can be activated from julia by executing 
`<julia> using Pkg>`
`<julia> Pkg.activate("./")>`
`<julia> using RaeDm>`


Building a model
----------------
A model can be built by calling the FIVOChain constructor:
`<FIVOChain(;nx::Int64=2,ny::Int64=2,nz::Int64=10,nlayers::Int64=0,nnodes::Int64=50,nsim::Int64=4,afun=elu)>`
where `<nx>` is the length of the regressor vector, `<ny>` should be 2 (RT and choices) and therefore left unchanged, `<nz>` is the size of the latent space.

`<nlayers>` is the number of layers of the Normalizing flow (if 0 no NF is used and the approximate posterior is spherical).

`<nnodes>` the number of nodes for both the feedforward layers and the RNN (GRU is used by default but LSTM could be used too without much effort), `<nsim>` the number of particles, and  `<afun>` is the activation function.

`<F = FIVOChain(nlayers=0,nx=length(X[1][1]),nsim=8)>` creates an instance of FIVOChain that can be used in the following way:
`<L = F(RT,C,X;
gradient_fetch_interval::Integer = -1, compute_intermediate_grad::Bool = false,opt_local=()->nothing,single_update::Bool=fa    lse, eval::Bool=false>`


`<gradient_fetch_interval>` indicates at which interval the gradient should be computed (i.e. where the likelihood has to be detached from past values of the latent variables and hidden layer). If <= 0, the gradient is computed without discontinuity.

`<compute_intermediate_grad>` indicates if intermediate gradient should be computed: in that case, the each time the likelihood is stacked from past values, a new gradient is computed.

`<opt_local>` is an optimisation function used by the optimizer.

`<single_update>` states that only one sub-gradient should be computed. This makes the optimization much faster for long traces without loosing much predictive power in practice. For instance, with a `<gradient_fetch_interval>` of 500 but a number of datapoints of 2000, only a random sequence of 500 trials will be used for the gradient computation and the other results will be considered as given during backpropagation.

`<eval>` indicates that the executable should return the `<FIVOChain>` structure with intermediate values of the particles stored in `<FIVOChain.output>`.

Passing data
------------
Data needs to be preprocessed as follow: first, for N trials, RT and choices need to be vectors (Nx1 matrices). X (the regressors) need to be a vector of vectors of size Nx1 x (Rx1), where R is the length of the regressor.

Training the algorithm
----------------------
The file in test/data/BugSa/run.jl gives a runnable example of how to train the RaeDm on a real dataset.
This dataset has 24 subjects executing a two-step TAFC task over 2000 trials. Because they were forced to give 2x2000 responses, there are at least 4000 trials for each subject.

The test/data/BugSa/load.jl script retrieves the data.

A FIVOChain object is build by

`<F = FIVOChain(nlayers=0,nx=length(X[1][1]),nsim=8)>`

Then an optimizer is created with

`<opt = RaeDm.optimize(Flux.ADAM(params(F), 0.0001))>`

And then trained with

`<opt = opt(F,RT,C,X,gradient_fetch_interval=interval,compute_intermediate_grad=true,single_update=true,continuous_opt=false)>`

After that, we can retrieve the simulated particled (weights and variables) by executing
```
 OPT.fc_out = []
 for ss in 1:length(RT)
   push!(OPT.fc_out, F(RT[ss],C[ss],X[ss],eval=true))
 end
```
