import StaticArrays
import StaticArrays: SVector, MVector
import JACC
import ArgParse

# @static if endswith(JACC.JACCPreferences.backend, "cuda")
# elseif endswith(JACC.JACCPreferences.backend, "amdgpu")
# end

be_verbose::Bool = false
function verbose(v = true)
    global be_verbose = v
    return nothing
end

@inline function lerp(a, b, t)
    return a + t * (b - a)
end

const SignalType = Float32
const CoordType = Float32
const ScalarType = Float32
const SizeType = Int

const Vector3{T} = StaticArrays.SVector{3,T} where {T}
@inline Vector3{T}() where {T} = Vector3{T}(0, 0, 0)
const Vector4{T} = StaticArrays.SVector{4,T} where {T}

const Vec3 = Vector3{ScalarType}
const V3 = StaticArrays.SA{ScalarType}
const Vec4 = Vector4{ScalarType}
const V4 = StaticArrays.SA{ScalarType}

const Crd3 = Vector3{CoordType}
const Crd4 = Vector4{CoordType}
const C3 = StaticArrays.SA{CoordType}
const C4 = StaticArrays.SA{CoordType}

const Id3 = Vector3{SizeType}
const I3 = StaticArrays.SA{SizeType}

const SquareMatrix3{T} = StaticArrays.SMatrix{3,3,T,9} where {T}
@inline SquareMatrix3{T}() where {T} = zeros(SquareMatrix3{T})

const SquareMatrix3r = SquareMatrix3{ScalarType}
const SquareMatrix3c = SquareMatrix3{CoordType}

const JACCArray = JACC.array_type()
const Array1 = JACCArray{T,1} where {T}
@inline Array1{T}(n::Int64) where {T} = Array1{T}(undef, n)
const Array1r = Array1{ScalarType}
const Array1c = Array1{CoordType}

const Array2 = JACCArray{T,2} where {T}
@inline Array2{T}(m::Int64, n::Int64) where {T} = Array2{T}(undef, m, n)
const Array2c = Array2{CoordType}

struct Optional{T}
    value::Union{T,Missing}
    Optional{T}() where {T} = new(missing)
    Optional{T}(x) where {T} = new(x)
    Optional{T}(::Missing) where {T} = new(missing)
end

ismissing(x::Optional) = x.value === missing
hasvalue(x::Optional) = !ismissing(x)
coalesce(x::Optional) = x.value
Base.convert(::Type{Optional{T}}, ::Missing) where {T} = Optional{T}()

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

@inline function partitionHistogramRange(r::AbstractRange)
    N = length(r) - 1
    binStart, binStop = getRankRange(N)
    valStart = r[binStart]
    valStop = r[binStop+1]
    nBins = binStop - binStart + 1
    nPts = nBins + 1
    return range(start = valStart, length = nPts, stop = valStop)
end

struct Options
    partition::String
    binmd::String
end

function parse_args(args::Vector{String})
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--partition", "-p"
            help = "MPI rank distribution target: files (default), histogram"
            arg_type = String
            default = "files"
        "--binmd", "-b"
            help = "BinMD implementation strategy: mantid (default), mantid1d columns, columns1d, boxes, boxes1d"
            arg_type = String
            default = "mantid"
    end
    pargs = ArgParse.parse_args(args, s)

    return Options(pargs["partition"], pargs["binmd"])
end
