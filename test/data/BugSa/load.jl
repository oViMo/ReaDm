using CSV,Printf
# csvs = Dict()
RT = Array{Array{Float64,1},1}()
C = Array{Array{Float64,1},1}()
X = Array{Array{Array{Float64,1}},1}()
for ss in 1:24
	fn = Printf.@sprintf("../BugSa_data/subj%i.csv",ss)
	csvs = CSV.read(fn)

	rt1 = csvs.RT1
	rt2 = csvs.RT2
	nanid = vec(hcat(isnan.(rt1),isnan.(rt2))')
	rt1[isnan.(rt1)] = csvs.TimeOut1[isnan.(rt1)]
	rt2[isnan.(rt2)] = csvs.TimeOut2[isnan.(rt2)]
	rt = vec(hcat(rt1,rt2)')
	c = vec(hcat(csvs.ChA1,csvs.ChA2)')

	x_tuple = (vec(hcat(csvs.A1,csvs.A2)'), # instructions
		   vec(hcat(csvs.Avoid,csvs.Avoid)), # avoided
		   vec(hcat(csvs.TO1,csvs.TO2)), # last time-out
		   vec(hcat(csvs.Prob1,csvs.Prob2)), # pred
		   vec(hcat(csvs.extinction,csvs.extinction)),
		   vec(hcat(one.(csvs.A1),2 .* one.(csvs.A2))')) # A1 or A2



	# delete trials where first response was nan
	iddel = findall(map((x,nanid)->x == 1 && nanid,x_tuple[end],nanid)) .+ 1
	deleteat!(rt[:],iddel)
	deleteat!(c[:],iddel)

	map(x->deleteat!(x[:],iddel),x_tuple)
	x = hcat(x_tuple...)

	push!(RT,rt)
	push!(C,c)
	push!(X,[x[k,:] for k in 1:size(x,1)])
end
