import StaticArrays: SMatrix, MVector
import Atomix

function binMD!(h::Hist3, events::AbstractArray, weights::AbstractArray, op::SquareMatrix3c)
    for n = 1:size(events, 1)
        v = op * C3[events[n, 1], events[n, 2], events[n, 3]]
        atomic_push!(h, v[1], v[2], v[3], weights[n])
    end
end

function binBoxes!(h::Hist3, events::FastEventData, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        (length(transforms), length(events.boxType)),
        (n, i, t) -> begin
            @inbounds begin
                eid = t.eventIndex
                eid2 = eid[i, 2]
                idxn = Iterators.drop(CartesianIndices((1:2, 3:4, 5:6)), 1)
                if t.boxType[i] == 1 && eid2 != 0
                    eid1 = eid[i, 1]
                    op = t.transforms[n]
                    if eid2 > 16
                        startIdx = MVector{3,SizeType}(0, 0, 0)
                        singleBox = true
                        vf = op * C3[t.extents[i, 1], t.extents[i, 3], t.extents[i, 5]]
                        for j = 1:3
                            startIdx[j] = binindex1d(t.h, j, vf[j])
                            if startIdx[j] < 1 || startIdx[j] > t.h.nbins[j]
                                singleBox = false
                                break
                            end
                        end
                        for k in idxn
                            if singleBox == false
                                break
                            end
                            vf = op * C3[t.extents[i, k[1]], t.extents[i, k[2]], t.extents[i, k[3]]]
                            for j = 1:3
                                endIdx = binindex1d(t.h, j, vf[j])
                                if startIdx[j] != endIdx
                                    singleBox = false
                                    break
                                end
                            end
                        end
                        if singleBox
                            Atomix.@atomic binweights(t.h)[startIdx...] += t.boxSignal[i, 1]
                        else
                            startId = eid1 + 1
                            stopId = startId + eid2 - 1
                            localevents = @view t.events[startId:stopId, :]
                            localweights = @view t.weights[startId:stopId]
                            binMD!(t.h, localevents, localweights, op)
                        end
                    else
                        startId = eid1 + 1
                        stopId = startId + eid2 - 1
                        localevents = @view t.events[startId:stopId, :]
                        localweights = @view t.weights[startId:stopId]
                        for n = 1:size(localevents, 1)
                            v = op * C3[localevents[n, 1], localevents[n, 2], localevents[n, 3]]
                            atomic_push!(h, v[1], v[2], v[3], localweights[n])
                        end
                    end
                end
            end
        end,
        (
            h = h,
            boxSignal = events.signal,
            events.boxType,
            events.events,
            events.weights,
            events.eventIndex,
            events.extents,
            transforms,
        ),
    )
end

function binBoxes1d!(h::Hist3, events::FastEventData, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        length(events.boxType),
        (i, t) -> begin
            @inbounds begin
                eid = t.eventIndex
                eid2 = eid[i, 2]
                idxn = Iterators.drop(CartesianIndices((1:2, 3:4, 5:6)), 1)
                if t.boxType[i] == 1 && eid2 != 0
                    eid1 = eid[i, 1]
                    if eid2 > 16
                        for op in t.transforms
                            vf = op * C3[t.extents[i, 1], t.extents[i, 3], t.extents[i, 5]]
                            startIdx = MVector{3,SizeType}(0, 0, 0)
                            singleBox = true
                            for j = 1:3
                                startIdx[j] = binindex1d(t.h, j, vf[j])
                                if startIdx[j] < 1 || startIdx[j] > t.h.nbins[j]
                                    singleBox = false
                                    break
                                end
                            end
                            for k in idxn
                                if singleBox == false
                                    break
                                end
                                vf = op * C3[t.extents[i, k[1]], t.extents[i, k[2]], t.extents[i, k[3]]]
                                for j = 1:3
                                    endIdx = binindex1d(t.h, j, vf[j])
                                    if startIdx[j] != endIdx
                                        singleBox = false
                                        break
                                    end
                                end
                            end
                            if singleBox
                                Atomix.@atomic binweights(t.h)[startIdx...] += t.boxSignal[i, 1]
                            else
                                startId = eid1 + 1
                                stopId = startId + eid2 - 1
                                localevents = @view t.events[startId:stopId, :]
                                localweights = @view t.weights[startId:stopId]
                                binMD!(t.h, localevents, localweights, op)
                            end
                        end
                    else
                        startId = eid1 + 1
                        stopId = startId + eid2 - 1
                        localevents = @view t.events[startId:stopId, :]
                        localweights = @view t.weights[startId:stopId]
                        for n = 1:size(localevents, 1)
                            col = C3[localevents[n, 1], localevents[n, 2], localevents[n, 3]]
                            weight = localweights[n]
                            for op in t.transforms
                                v = op * col
                                atomic_push!(h, v[1], v[2], v[3], weight)
                            end
                        end
                    end
                end
            end
        end,
        (
            h = h,
            boxSignal = events.signal,
            events.boxType,
            events.events,
            events.weights,
            events.eventIndex,
            events.extents,
            transforms,
        ),
    )
end

