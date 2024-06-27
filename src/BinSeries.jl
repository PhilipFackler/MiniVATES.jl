import MPI
using Printf

@inline function getRankRange(N::Integer)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    count = trunc(Int, N / size)
    remainder = trunc(Int, N % size)
    if rank < remainder
        # The first 'remainder' ranks get 'count + 1' tasks each
        start = rank * (count + 1)
        stop = start + count
    else
        # The remaining 'size - remainder' ranks get 'count' task each
        start = rank * count + remainder
        stop = start + (count - 1)
    end

    return (start + 1, stop + 1)
end

tmfmt(tm::AbstractFloat) = @sprintf("%3.6f", tm)

@inline function binSeries!(
    signal::Hist3,
    eventsHist::Hist3,
    mdn::MDNorm,
    saFile::AbstractString,
    fluxFile::AbstractString,
    eventFilePairs::Vector{NTuple{2,AbstractString}},
    m_W::SquareMatrix3c,
)
    saData = loadSolidAngleData(saFile)
    fluxData = loadFluxData(fluxFile)

    exFile, eventFile = first(eventFilePairs)
    exData = loadExtrasData(exFile)
    setExtrasData!(mdn, exData)
    eventData = loadEventData(eventFile)

    set_m_W!(exData, m_W)
    transforms2 = makeTransforms(exData)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    commSize = MPI.Comm_size(comm)
    nFiles = length(eventFilePairs)
    start, stop = getRankRange(nFiles)

    if MiniVATES.be_verbose
        if rank == 0
            println("number of files: ", nFiles)
        end
        @show rank, start, stop
    end

    updAvg = 0.0
    mdnAvg = 0.0
    binAvg = 0.0

    for fi = start:stop
        exFile, eventFile = eventFilePairs[fi]
        let extrasWS = ExtrasWorkspace(exFile)
            exData.rotMatrix = getRotationMatrix(extrasWS)
        end

        updateEventsTime = nothing
        let eventWS = EventWorkspace(eventFile)
            eventData.protonCharge = getProtonCharge(eventWS)
            dur = @elapsed updateEvents!(eventData, eventWS)
            updateEventsTime = dur
            updAvg += dur
        end

        transforms = makeRotationTransforms(exData)

        dur = @elapsed mdNorm!(signal, mdn, saData, fluxData, eventData, transforms)
        mdNormTime = dur
        mdnAvg += dur

        dur = @elapsed binEvents!(eventsHist, eventData.events, transforms2)
        binEventsTime = dur
        binAvg += dur

        for r = 0:(commSize - 1)
            if rank == r
                println(
                    "rank: ",
                    lpad(rank, 2),
                    "; fi: ",
                    lpad(fi, 3),
                    "; updateEvents: ",
                    tmfmt(updateEventsTime),
                    "s, mdNorm: ",
                    tmfmt(mdNormTime),
                    "s, binEvents: ",
                    tmfmt(binEventsTime),
                    "s",
                )
            end
            MPI.Barrier(comm)
        end
    end
    sum = MPI.Reduce((updAvg, mdnAvg, binAvg), .+, comm)
    if rank == 0
        avg = sum ./ nFiles
        println("Averages:")
        println("    updateEvents: ", tmfmt(avg[1]), "s")
        println("    mdNorm:       ", tmfmt(avg[2]), "s")
        println("    binEvents:    ", tmfmt(avg[3]), "s")
        println()
    end

    return (signal, eventsHist)
end

function mergeHistogramToRootProcess(hist::Hist3)
    weights = MPI.Reduce(Core.Array(binweights(hist)), .+, MPI.COMM_WORLD)
    x, y, z = edges(hist)
    return Hist3(
        (Core.Array(x), Core.Array(y), Core.Array(z)),
        nbins(hist),
        origin(hist),
        boxLength(hist),
        weights,
    )
end
