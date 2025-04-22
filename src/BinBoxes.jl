import Adapt

function binBoxes!(h::Hist3, events::FastEventData, transforms::AbstractArray)
    JACC.parallel_for(
        length(events.boxType),
        (i, t) -> begin
            @inbounds begin
	        if t.boxType[i] == 1 && t.eventIndex[i,1] != 0
                end
            end		
        end,
        (h , boxType = events.boxType, eventIndex = events.eventIndex, transforms),
    )
end

