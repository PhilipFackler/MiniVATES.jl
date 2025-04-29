import Adapt

function binEvents!(h::Hist3, events::AbstractArray, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        (length(transforms), size(events, 2)),
        (n, i, t) -> begin
            @inbounds begin
                op = t.transforms[n]
                v = op * C3[t.events[6, i], t.events[7, i], t.events[8, i]]
                atomic_push!(t.h, v[1], v[2], v[3], t.events[1, i])
            end
        end,
        (h = h, events, transforms),
    )
end

function binEvents1d!(h::Hist3, events::AbstractArray, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        size(events, 2),
        (i, t) -> begin
            @inbounds begin
                event = C3[t.events[6, i], t.events[7, i], t.events[8, i]]
		weight = t.events[1, i]
		for n in 1:t.t_length
		    op = t.transforms[n]
                    v = op * event
                    atomic_push!(t.h, v[1], v[2], v[3], weight)
                end
            end		
        end,
	(h = h, events, transforms, t_length = length(transforms)),
    )
end

function binEvents!(h::Hist3, events::AbstractArray, weights::AbstractArray, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        (length(transforms), size(events, 1)),
        (n, i, t) -> begin
            @inbounds begin
		event = C3[t.events[i, 1], t.events[i, 2], t.events[i, 3]]
                op = t.transforms[n]
                v = op * event
                atomic_push!(t.h, v[1], v[2], v[3], t.weights[i])
            end
        end,
        (h = h, events, weights, transforms),
    )
end

function binEvents1d!(h::Hist3, events::AbstractArray, weights::AbstractArray, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        size(events, 1),
        (i, t) -> begin
            @inbounds begin
                event = C3[t.events[i, 1], t.events[i, 2], t.events[i, 3]]
                weight = t.weights[i]
                for n in 1:t.t_length
                    op = t.transforms[n]
                    v = op * event
                    atomic_push!(t.h, v[1], v[2], v[3], weight)
                end
            end
        end,
        (h, events, weights, transforms, t_length = length(transforms)),
    )
end

