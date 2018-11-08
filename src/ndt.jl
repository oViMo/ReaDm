# NDT utilities
function boundτ(L,τ,RT)
	if τ<RT
		return L
	else
		return -1e10*one(L)
	end
end
boundτ(L::TrackedReal,τ,RT) = Tracker.track(boundτ,L,τ,RT)
@grad function boundτ(L,τ,RT)
	if τ<RT
		return data(L),∇->(∇,∇*0,∇*0)
	else
		return -1e10,∇->(∇,∇*(RT-τ),∇*(τ-RT))
	end
end
