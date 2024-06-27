import Atomix
import Adapt
import Base: @propagate_inbounds

struct Hist3{TArray1,TArray3}
    edges::NTuple{3,TArray1}
    nbins::NTuple{3,SizeType}
    origin::Vector3{CoordType}
    boxLength::Vector3{CoordType}
    weights::TArray3
end

function Hist3(x::AbstractArray, y::AbstractArray, z::AbstractArray)
    nbins = (length(x) - 1, length(y) - 1, length(z) - 1)
    Hist3(
        (Array1c(x), Array1c(y), Array1c(z)),
        nbins,
        V3[x[1], y[1], z[1]],
        V3[x[2] - x[1], y[2] - y[1], z[2] - z[1]],
        JACC.zeros(SignalType, nbins),
    )
end

Adapt.@adapt_structure Hist3

@inline edges(h::Hist3) = h.edges

@inline nbins(h::Hist3) = h.nbins

@inline origin(h::Hist3) = h.origin

@inline boxLength(h::Hist3) = h.boxLength

@inline binweights(h::Hist3) = h.weights

@inline function reset!(h::Hist3)
    fill!(h.weights, 0.0)
end

@propagate_inbounds function binindex1d(h::Hist3, d::SizeType, crd::CoordType)
    dist = crd - h.origin[d]
    if dist < 0.0
        return 0
    end

    idx = trunc(Int, dist / h.boxLength[d]) + 1
    if idx > h.nbins[d]
        return h.nbins[d] + 1
    end

    return idx
end

@propagate_inbounds function binindex(h::Hist3, x, y, z)
    return (
        binindex1d(h, 1, convert(CoordType, x)),
        binindex1d(h, 2, convert(CoordType, y)),
        binindex1d(h, 3, convert(CoordType, z)),
    )
end

@propagate_inbounds function atomic_push!(
    h::Hist3,
    x::CoordType,
    y::CoordType,
    z::CoordType,
    wt,
)
    ix, iy, iz = binindex(h, x, y, z)
    lx, ly, lz = h.nbins
    if (unsigned(ix - 1) < lx) && (unsigned(iy - 1) < ly) && (unsigned(iz - 1) < lz)
        Atomix.@atomic h.weights[ix, iy, iz] += wt
    end
end

