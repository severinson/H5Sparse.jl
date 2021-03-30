module H5Sparse

using HDF5, SparseArrays

export H5SparseMatrixCSC

"""
    H5SparseMatrixCSC{Tv, Ti<:Integer} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}

Out-of-core `AbstractSparseMatrixCSC` backed by a HDF5 dataset stored on disk.

```julia
# Conversion from SparseMatrixCSC; writes B to a dataset "A" in the file "foo.h5"
using SparseArrays
B = sprand(10, 10, 0.5)
A = H5SparseMatrixCSC("foo.h5", "A", B)

# or, equivalently
using HDF5
fid = h5open("foo.h5", "cw")
A = H5SparseMatrixCSC(fid, "A", B)

# kwargs are passed on to h5writecsc
C = sprand(10, 10, 0.5)
A = H5SparseMatrixCSC("foo.h5", "A", C, overwrite=true) # Overwrites any existing dataset with name A

# Append a SparseMatrixCSC to the right; useful for constructing large matrices in an iterative fashion
D = sprand(10, 5, 0.5)
append!(A, D)       # A is now of size (10, 15)

# Reading the entire matrix from disk
sparse(A)           # SparseMatrixCSC
Matrix(A)           # Matrix

# Reading querying columns, or blocks of columns, is fast, quering rows is slow
@time A[:, 1];      # 0.000197 seconds (77 allocations: 3.844 KiB)
@time A[:, 1:10];   # 0.000192 seconds (71 allocations: 4.234 KiB)
@time A[1, :];      # 0.001479 seconds (1.35 k allocations: 69.109 KiB)
```

"""
struct H5SparseMatrixCSC{Tv, Ti<:Integer} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
    fid::HDF5.File  # Backing HDF5 file
    name::String    # Dataset name, i.e., data is stored in fid[name]
    function H5SparseMatrixCSC{Tv,Ti}(fid::HDF5.File, name::AbstractString) where {Tv,Ti<:Integer}
        name in keys(fid) || throw(ArgumentError("$name is not in $fid"))
        g = fid[name]
        "m" in keys(g) || throw(ArgumentError("m is not in $g"))
        "n" in keys(g) || throw(ArgumentError("n is not in $g"))
        "colptr" in keys(g) || throw(ArgumentError("colptr is not in $g"))
        "rowval" in keys(g) || throw(ArgumentError("rowval is not in $g"))
        "nzval" in keys(g) || throw(ArgumentError("nzval is not in $g"))
        new{Tv,Ti}(fid, String(name))
    end
end
H5SparseMatrixCSC(filename::AbstractString, name::AbstractString, B::SparseMatrixCSC; kwargs...) = H5SparseMatrixCSC(h5open(filename, "cw"), name, B; kwargs...)
function H5SparseMatrixCSC(fid::HDF5.File, name::AbstractString, B::SparseMatrixCSC{Tv,Ti}; kwargs...) where {Tv,Ti}
    h5writecsc(fid, name, B; kwargs...)
    H5SparseMatrixCSC{Tv,Ti}(fid, name)
end

function Base.show(io::IO, A::H5SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    print(io, "H5SparseMatrixCSC{$Tv,$Ti}($(A.fid.filename), $(A.name))")
end

function Base.display(A::H5SparseMatrixCSC)    
    show(stdout, A)
end

SparseArrays.getcolptr(A::H5SparseMatrixCSC) = A.fid[A.name]["colptr"]
SparseArrays.rowvals(A::H5SparseMatrixCSC) = A.fid[A.name]["rowval"]
SparseArrays.nonzeros(A::H5SparseMatrixCSC) = A.fid[A.name]["nzval"]

Base.size(A::H5SparseMatrixCSC) = h5size(A.fid, A.name)
function Base.eltype(A::H5SparseMatrixCSC{Tv}) where Tv
    Tv
end

struct H5View{Tv} <: AbstractVector{Tv}
    x::HDF5.Dataset # Data    
    i::Int          # Lower index
    j::Int          # Upper index
end
H5View(x::HDF5.Dataset, i::Integer, j::Integer) = H5View{eltype(x)}(x, i, j)

function Base.show(io::IO, a::H5View{Tv}) where Tv
    print(io, "H5View{$Tv}($(a.x), $(x.i), $(x.j))")
end

function Base.display(a::H5View)    
    show(stdout, a)
end

Base.length(a::H5View) = a.j-a.i+1
Base.size(a::H5View) = (length(a),)
function Base.eltype(::H5View{Tv}) where Tv
    Tv
end
function Base.getindex(a::H5View{Tv}, k::Integer)::Tv where Tv
    @boundscheck 0 < k <= length(a)
    a.x[a.i+k-1]
end

function Base.getindex(A::H5SparseMatrixCSC{Tv}, row::Integer, col::Integer)::Tv where Tv
    m, n = size(A)    
    @boundscheck 0 < row <= m || throw(BoundsError(A, row))
    @boundscheck 0 < col <= n || throw(BoundsError(A, col))
    g = A.fid[A.name]    
    i = g["colptr"][col]
    j = g["colptr"][col+1] - 1
    v = H5View(g["rowval"], i, j)
    k = searchsortedfirst(v, row) + i - 1
    if k > j || g["rowval"][k] != row
        return zero(Tv)
    else
        return g["nzval"][k]
    end
end

function Base.getindex(A::H5SparseMatrixCSC{Tv,Ti}, row::Integer, ::Colon)::SparseVector{Tv,Ti} where {Tv,Ti}
    m, n = size(A)    
    @boundscheck 0 < row <= m || throw(BoundsError(A, row))
    nzind = zeros(Ti, 0)
    nzval = zeros(Tv, 0)
    for col in 1:n
        v = A[row, col]
        if !iszero(v)
            push!(nzind, Ti(col))
            push!(nzval, v)
        end
    end
    SparseVector{Tv,Ti}(n, nzind, nzval)
end

function Base.getindex(A::H5SparseMatrixCSC{Tv,Ti}, ::Colon, col::Integer)::SparseVector{Tv,Ti} where {Tv,Ti}
    m, n = size(A)
    @boundscheck 0 < col <= n || throw(BoundsError(A, col))
    g = A.fid[A.name]    
    i = g["colptr"][col]
    j = g["colptr"][col+1] - 1
    rowval = g["rowval"][i:j]
    nzval = g["nzval"][i:j]
    SparseVector{Tv,Ti}(m, rowval, nzval)
end

"""

Read the submatrix consisting of columns `firstcol:lastcol` from `fid[name]`.
"""
function Base.getindex(A::H5SparseMatrixCSC{Tv,Ti}, ::Colon, cols::UnitRange{<:Integer})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}
    m, n = size(A)
    firstcol, lastcol = first(cols), last(cols)    
    @boundscheck 0 < firstcol <= n || throw(BoundsError(A, firstcol))
    @boundscheck 0 < lastcol <= n || throw(BoundsError(A, lastcol))
    @boundscheck firstcol <= lastcol <= n || throw(ArgumentError("first column is $firstcol, but last column is $lastcol"))
    g = A.fid[A.name]
    colptr = g["colptr"][firstcol:lastcol+1]
    i = colptr[1]
    j = colptr[end] - 1
    rowval = g["rowval"][i:j]     
    nzval = g["nzval"][i:j]
    colptr .-= i-1
    SparseMatrixCSC{Tv,Ti}(m, lastcol-firstcol+1, colptr, rowval, nzval)
end

"""
    append!(A::H5SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Appends `B` to the right of `A`.
"""
function Base.append!(A::H5SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}    
    size(A, 1) == size(B, 1) || throw(DimensionMismatch("A has dimensions $(size(A)), but B has dimensions $(size(B))"))
    h5appendcsc(A.fid, A.name, B)
    A
end

function SparseArrays.sparse(A::H5SparseMatrixCSC{Tv,Ti})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}
    m, n = size(A)    
    g = A.fid[A.name]
    colptr = g["colptr"][:]    
    rowval = g["rowval"][:]    
    nzval = g["nzval"][:]
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function h5size(fid::HDF5.File, name::AbstractString)
    g = fid[name]
    g["m"][], g["n"][]
end

"""
    h5writecsc(fid::HDF5.File, name::AbstractString, B::SparseMatrixCSC; kwargs...) 

Write `B` to `fid[name]`, overwriting any existing dataset with name `name` if `overwrite=true`. 
This is a low-level routine without error checking; user should typically use `H5SparseMatrixCSC`
instead.
"""
function h5writecsc(fid::HDF5.File, name::AbstractString, B::SparseMatrixCSC; kwargs...) 
    m, n = size(B)
    h5writecsc(fid, name, m, n, SparseArrays.getcolptr(B), rowvals(B), nonzeros(B); kwargs...)
end
function h5writecsc(fid::HDF5.File, name::AbstractString, m::Integer, n::Integer, colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer}, nzval::AbstractVector; overwrite=false, chunk=nothing, blosc=5, kwargs...)
    if name in keys(fid)
        if overwrite
            delete_object(fid, name)
        else
            throw(ArgumentError("$name already exists in $fid"))
        end
    end
    g = create_group(fid, name)
    g["m"] = m
    g["n"] = n    
    create_dataset(
        g, "colptr", 
        eltype(colptr), 
        ((length(colptr),), (-1,)),
        chunk=isnothing(chunk) ? HDF5.heuristic_chunk(colptr) : chunk,
        blosc=blosc,
        kwargs...,
    )[1:length(colptr)] = colptr
    create_dataset(
        g, "rowval", 
        eltype(rowval), 
        ((length(rowval),), (-1,)),
        chunk=isnothing(chunk) ? HDF5.heuristic_chunk(rowval) : chunk,
        blosc=blosc,
        kwargs...,
    )[1:length(rowval)] = rowval    
    create_dataset(
        g, "nzval",
        eltype(nzval), 
        ((length(nzval),), (-1,)),
        chunk=isnothing(chunk) ? HDF5.heuristic_chunk(nzval) : chunk,
        blosc=blosc,
        kwargs...,
    )[1:length(nzval)] = nzval
    return
end

"""
    h5appendcsc(fid::HDF5.File, name::AbstractString, B::SparseMatrixCSC)

Append `B` to the right of the `SparseMatrixCSC` stored in `fid[name]`. This is a low-level routine
without error checking; user should typically use `append` instead.
"""
function h5appendcsc(fid::HDF5.File, name::AbstractString, B::SparseMatrixCSC)
    m, n = size(B)
    h5appendcsc(fid, name, m, n, SparseArrays.getcolptr(B), rowvals(B), nonzeros(B))
end
function h5appendcsc(fid::HDF5.File, name::AbstractString, m::Integer, n::Integer, colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer}, nzval::AbstractVector)
    g = fid[name]
    # colptr
    i = size(g["colptr"], 1)
    j = i + length(colptr) - 1
    HDF5.set_extent_dims(g["colptr"], (j,))
    offset = size(g["rowval"], 1)
    g["colptr"][i:j]  = colptr .+ offset
    # rowval
    i = size(g["rowval"], 1) + 1
    j = i + length(rowval) - 1
    HDF5.set_extent_dims(g["rowval"], (j,))
    g["rowval"][i:j]  = rowval
    # nzval    
    i = size(g["nzval"], 1) + 1
    j = i + length(nzval) - 1
    HDF5.set_extent_dims(g["nzval"], (j,))
    g["nzval"][i:j]  = nzval
    # n
    old_n = g["n"][]
    delete_object(g, "n")
    g["n"] = old_n + n
    return
end

end