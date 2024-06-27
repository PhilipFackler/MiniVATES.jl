include("test_data_constants.jl")
include("common.jl")

import MiniVATES
import MiniVATES: Hist3, C3

import MPI

dur = @elapsed begin
    x = range(start = -7.5375, length = 604, stop = 7.5375)
    y = range(start = -13.16524, length = 604, stop = 13.16524)
    z = range(start = -0.5, length = 2, stop = 0.5)

    signal = Hist3(x, y, z)
    h = Hist3(x, y, z)
    doctest = MiniVATES.MDNorm(signal)

    extras_events_files = Vector{NTuple{2,AbstractString}}()
    for file_num = benzil_event_nxs_min:benzil_event_nxs_max
        fNumStr = string(file_num)
        exFile = benzil_event_nxs_prefix * fNumStr * "_extra_params.hdf5"
        eventFile = benzil_event_nxs_prefix * fNumStr * "_BEFORE_MDNorm.nxs"
        push!(extras_events_files, (exFile, eventFile))
    end

    # MiniVATES.verbose()
    MiniVATES.binSeries!(
        signal,
        h,
        doctest,
        benzil_sa_nxs_file,
        benzil_flux_nxs_file,
        extras_events_files,
        C3[1.0 1.0 0.0; 1.0 -1.0 0.0; 0.0 0.0 1.0],
    )

    signalMerged = MiniVATES.mergeHistogramToRootProcess(signal)
    hMerged = MiniVATES.mergeHistogramToRootProcess(h)
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    write_cat(signalMerged, hMerged)
    println("Total app time: ", dur, "s")
end
