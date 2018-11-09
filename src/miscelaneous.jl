
function overflow_prev(x)
	V=1e10*one(x)
	return -V<x<V
end
function logsumexp_overflow(x)
	if any(overflow_prev,x)
		return logsumexp_overflow_naive(x)
	else
		return -floatmax()
	end
end
function logsumexp_overflow_naive(x)
	mx=all(overflow_prev,x) ? maximum(x) : begin
		maximum(x[overflow_prev.(x)])
	end
	L = zero(x[1])
	for k in eachindex(x)
		L += overflow_prev(x[k]) ? exp(x[k]-mx) : x[k]<0 ? zero(x[k]) : floatmax()
	end
	L = log(L)+mx
	return L
end
logsumexp_overflow(x::TrackedArray) = Tracker.track(logsumexp_overflow,x)
@grad function logsumexp_overflow(x_diff)
	x = data(x_diff)
	if any(overflow_prev,x)
		mx=all(overflow_prev,x) ? maximum(x) : begin
			maximum(x[overflow_prev.(x)])
		end
		L=zero(x[1])
		∂ = similar(x)
		for k in eachindex(x)
			L+=overflow_prev(x[k]) ? (ex = exp(x[k]-mx)) : x[k]<0 ? (ex = zero(x[k])) : begin
				ex = 0.0
				floatmax()
			end
			∂[k]=ex
		end
		∂ ./= L
		L = log(L)+mx
		return elinf(L),Δ->(Δ*∂,)
	else
		∂ = one.(x)/length(x)
		return -floatmax(),Δ->(Δ*∂,)
	end
end

elinf(L) = L==-Inf ? -floatmax() : L
@grad function elinf(L)
	d = data(L)
	return elinf(d),Δ->(Δ,)
end
elinf(x::TrackedReal) = Tracker.track(elinf,x)
