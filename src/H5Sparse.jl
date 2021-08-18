"""

Support for out-of-core sparse arrays backed by a HDF5 file stored on disk. Provides `H5SparseMatrixCSC`.
"""
module H5Sparse

using HDF5, SparseArrays

export H5SparseMatrixCSC

"""
    H5SparseMatrixCSC{Tv, Ti<:Integer, Td<:HDF5.H5DataStore} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}

Out-of-core `AbstractSparseMatrixCSC` backed by a dataset stored on disk, of type `Td<:HDF5.H5DataStore`, e.g., a HDF5 file.

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
A = H5SparseMatrixCSC("foo.h5", "A", B, overwrite=true) # Overwrites any existing dataset with name A

# Construct from an existing file
A = H5SparseMatrixCSC("foo.h5", "A")
A = H5SparseMatrixCSC(fid, "A")

# Construct a view into a subset of the rows and/or columns stored in a file
A = H5SparseMatrixCSC("foo.h5", "A", :, 2:5)

# Colon or UnitRange indexing returns a new H5SparseMatrixCSC that is a view into the specified subset of rows and/or columns
A[:, 1:10]
A[1:4, :]

# Integer indexing returns the requested element
A[1, 1]

# Concatenate with a SparseMatrixCSC to the right; useful for constructing large matrices in an iterative fashion
# Returns a new H5SparseMatrixCSC spanning all columns of the resulting matrix
C = sprand(10, 5, 0.5)
A = hcat(A, C)      # A is now of size (10, 15)

# Use sparse or Matrix to load H5SparseMatrixCSC matrix into memory
sparse(A)           # SparseMatrixCSC
Matrix(A)           # Matrix
```

"""

struct H5SparseMatrixCSC{Tv, Ti<:Integer, Td<:HDF5.H5DataStore} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
    fid::Td                 # Backing storage
    name::String            # Dataset name, i.e., data is stored in fid[name]
    rows::UnitRange{Int}    # Subset of rows stored in fid[name] accessible via this instance
    cols::UnitRange{Int}    # Subset of columns stored in fid[name] accessible via this instance
    function H5SparseMatrixCSC(fid::HDF5.H5DataStore, name::AbstractString, rows::UnitRange{Int}, cols::UnitRange{Int}) where {Tv,Ti<:Integer}
        name in keys(fid) || throw(ArgumentError("$name is not in $fid"))
        g = fid[name]
        g isa HDF5.Group || throw(ArgumentError("fid[name] is $g, but must be a HDF5.Group"))
        "m" in keys(g) || throw(ArgumentError("m is not in $g"))
        "n" in keys(g) || throw(ArgumentError("n is not in $g"))
        "colptr" in keys(g) || throw(ArgumentError("colptr is not in $g"))
        "rowval" in keys(g) || throw(ArgumentError("rowval is not in $g"))
        "nzval" in keys(g) || throw(ArgumentError("nzval is not in $g"))
        length(size(g["colptr"])) == 1 || return false
        length(size(g["rowval"])) == 1 || return false
        length(size(g["nzval"])) == 1 || return false        
        eltype(g["colptr"]) == eltype(g["rowval"]) || throw(ArgumentError("colptr has eltype $(g["colptr"])), but rowval has eltype $(g["rowval"]))"))
        m, n = g["m"][], g["n"][]
        0 < first(rows) <= m || throw(ArgumentError("first row is $(first(rows)), but m is $m"))
        0 < last(rows) <= m || throw(ArgumentError("last row is $(last(rows)), but m is $m"))
        first(rows) <= last(rows) || throw(ArgumentError("first row is $(first(rows)), but last row is $(last(rows))"))
        0 < first(cols) <= n || throw(ArgumentError("first column is $(first(cols)), but n is $n"))
        0 < last(cols) <= n || throw(ArgumentError("last column is $(last(cols)), but n is $n"))
        first(cols) <= last(cols) || throw(ArgumentError("first columns is $(first(cols)), but last column is $(last(cols))"))
        new{eltype(g["nzval"]),eltype(g["rowval"]),typeof(fid)}(fid, String(name), rows, cols)
    end
end
H5SparseMatrixCSC(filename::AbstractString, args...; kwargs...) = H5SparseMatrixCSC(h5open(filename, "cw"), args...; kwargs...)
function H5SparseMatrixCSC(fid::HDF5.H5DataStore, name::AbstractString, B::SparseMatrixCSC; kwargs...)
    h5writecsc(fid, name, B; kwargs...)
    H5SparseMatrixCSC(fid, name, 1:size(B, 1), 1:size(B, 2))
end
function H5SparseMatrixCSC(fid::HDF5.H5DataStore, name::AbstractString)
    m, n = h5size(fid, name)
    H5SparseMatrixCSC(fid, name, 1:m, 1:n)
end
function H5SparseMatrixCSC(fid::HDF5.H5DataStore, name::AbstractString, ::Colon, cols)
    m, n = h5size(fid, name)
    H5SparseMatrixCSC(fid, name, 1:m, cols)
end
function H5SparseMatrixCSC(fid::HDF5.H5DataStore, name::AbstractString, rows, ::Colon)
    m, n = h5size(fid, name)
    H5SparseMatrixCSC(fid, name, rows, 1:n)
end
H5SparseMatrixCSC(fid::HDF5.H5DataStore, name::AbstractString, ::Colon, ::Colon) = H5SparseMatrixCSC(fid, name)

Base.show(io::IOContext, A::H5SparseMatrixCSC) = show(io.io, A)

function Base.show(io::IO, A::H5SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    print(io, "H5SparseMatrixCSC{$Tv,$Ti}($(A.fid.filename), $(A.name), $(A.rows), $(A.cols))")
end

function Base.display(A::H5SparseMatrixCSC)    
    show(stdout, A)
end

Base.size(A::H5SparseMatrixCSC) = (last(A.rows)-first(A.rows)+1, last(A.cols)-first(A.cols)+1)
function Base.eltype(::H5SparseMatrixCSC{Tv}) where Tv
    Tv
end

struct H5View{Tv} <: AbstractVector{Tv}
    x::HDF5.Dataset # Data    
    i::Int          # Lower index
    j::Int          # Upper index
end
H5View(x::HDF5.Dataset, i::Integer, j::Integer) = H5View{eltype(x)}(x, i, j)
H5View(x::HDF5.Dataset) = H5View(x, 1, length(x))

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

SparseArrays.getcolptr(A::H5SparseMatrixCSC) = H5View(A.fid[A.name]["colptr"])
SparseArrays.rowvals(A::H5SparseMatrixCSC) = H5View(A.fid[A.name]["rowval"])
SparseArrays.nonzeros(A::H5SparseMatrixCSC) = H5View(A.fid[A.name]["nzval"])

function Base.getindex(A::H5SparseMatrixCSC{Tv}, row::Integer, col::Integer)::Tv where Tv
    m, n = size(A)    
    @boundscheck 0 < row <= m || throw(BoundsError(A, (row, col)))
    @boundscheck 0 < col <= n || throw(BoundsError(A, (row, col)))
    row = first(A.rows) + row - 1
    col = first(A.cols) + col - 1
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

Base.getindex(A::H5SparseMatrixCSC, ::Colon, cols::UnitRange{<:Integer}) = getindex(A, 1:size(A, 1), cols)
Base.getindex(A::H5SparseMatrixCSC, rows::UnitRange{<:Integer}, ::Colon) = getindex(A, rows, 1:size(A, 2))
Base.getindex(A::H5SparseMatrixCSC, ::Colon, ::Colon) = getindex(A, 1:size(A, 1), 1:size(A, 2))
function Base.getindex(A::H5SparseMatrixCSC{Tv,Ti}, rows::UnitRange{<:Integer}, cols::UnitRange{<:Integer})::H5SparseMatrixCSC{Tv,Ti} where {Tv,Ti}
    m, n = size(A)  
    @boundscheck 0 < first(rows) <= m || throw(BoundsError(A, (rows, cols)))
    @boundscheck 0 < last(rows) <= m || throw(BoundsError(A, (rows, cols)))
    @boundscheck first(rows) <= last(rows) || throw(ArgumentError("first row is $(first(rows)), but last row is $(last(rows))"))        
    @boundscheck 0 < first(cols) <= n || throw(BoundsError(A, (rows, cols)))
    @boundscheck 0 < last(cols) <= n || throw(BoundsError(A, (rows, cols)))
    @boundscheck first(cols) <= last(cols) || throw(ArgumentError("first column is $(first(cols)), but last column is $(last(cols))"))
    H5SparseMatrixCSC(A.fid, A.name, (first(A.rows)+first(rows)-1):(first(A.rows)+last(rows)-1), (first(A.cols)+first(cols)-1):(first(A.cols)+last(cols)-1))
end

function Base.setindex!(::H5SparseMatrixCSC, ::Any, ::Integer, ::Integer)
    error("H5SparseMatrixCSC data is immutable")
end

"""
    append!(A::H5SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Appends `B` to the right of `A`.
"""
function Base.hcat(A::H5SparseMatrixCSC{Tv,Ti}, Bs::SparseMatrixCSC{Tv,Ti}...) where {Tv,Ti}
    ncols = 0
    for B in Bs
        size(A, 1) == size(B, 1) || throw(DimensionMismatch("A has dimensions $(size(A)), but B has dimensions $(size(B))"))
        h5appendcsc(A.fid, A.name, B)
        ncols += size(B, 2)
    end
    H5SparseMatrixCSC(A.fid, A.name, A.rows, first(A.cols):(last(A.cols)+ncols))
end

function SparseArrays.sparse(A::H5SparseMatrixCSC{Tv,Ti})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}
    m, n = size(A)
    if m != h5size(A.fid, A.name)[1]
        error("not implemented")
    end
    g = A.fid[A.name]
    colptr = g["colptr"][first(A.cols):(last(A.cols)+1)]
    i = colptr[1]
    j = colptr[end] - 1
    rowval = g["rowval"][i:j]
    nzval = g["nzval"][i:j]
    colptr .-= i-1
    SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

function h5size(fid::HDF5.H5DataStore, name::AbstractString)
    g = fid[name]
    g["m"][], g["n"][]
end

"""
    h5writecsc(fid::HDF5.H5DataStore, name::AbstractString, B::SparseMatrixCSC; kwargs...) 

Write `B` to `fid[name]`, overwriting any existing dataset with name `name` if `overwrite=true`. 
This is a low-level routine without error checking; user should typically use `H5SparseMatrixCSC`
instead.
"""
function h5writecsc(fid::HDF5.H5DataStore, name::AbstractString, B::SparseMatrixCSC; kwargs...) 
    m, n = size(B)
    h5writecsc(fid, name, m, n, SparseArrays.getcolptr(B), rowvals(B), nonzeros(B); kwargs...)
end
function h5writecsc(fid::HDF5.H5DataStore, name::AbstractString, m::Integer, n::Integer, colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer}, nzval::AbstractVector; overwrite=false, colptr_chunk=length(colptr) > 0 ? HDF5.heuristic_chunk(colptr) : [1], val_chunk=length(rowval) > 0 ? HDF5.heuristic_chunk(rowval) : [1], blosc=5, kwargs...)
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
    ds = create_dataset(
        g, "colptr", 
        eltype(colptr), 
        ((length(colptr),), (-1,)),
        chunk=colptr_chunk,
        blosc=blosc,
        kwargs...,
    )
    if length(colptr) > 0
        ds[1:length(colptr)] = colptr
    end
    ds = create_dataset(
        g, "rowval", 
        eltype(rowval), 
        ((length(rowval),), (-1,)),
        chunk=val_chunk,
        blosc=blosc,
        kwargs...,
    )
    if length(rowval) > 0
        ds[1:length(rowval)] = rowval    
    end
    ds = create_dataset(
        g, "nzval",
        eltype(nzval), 
        ((length(nzval),), (-1,)),
        chunk=val_chunk,
        blosc=blosc,
        kwargs...,
    )
    if length(nzval) > 0
        ds[1:length(nzval)] = nzval
    end
    return
end

"""
    h5appendcsc(fid::HDF5.H5DataStore, name::AbstractString, B::SparseMatrixCSC)

Append `B` to the right of the `SparseMatrixCSC` stored in `fid[name]`. This is a low-level routine
without error checking; user should typically use `append` instead.
"""
function h5appendcsc(fid::HDF5.H5DataStore, name::AbstractString, B::SparseMatrixCSC)
    m, n = size(B)
    h5appendcsc(fid, name, m, n, SparseArrays.getcolptr(B), rowvals(B), nonzeros(B))
end
function h5appendcsc(fid::HDF5.H5DataStore, name::AbstractString, m::Integer, n::Integer, colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer}, nzval::AbstractVector)
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

"""

Return `true` if `fid[name]` is a valid `H5SparseMatrixCSC` dataset, and `false` otherwise.
"""
function h5isvalidcsc(fid::HDF5.H5DataStore, name::AbstractString)
    name in keys(fid) || return false
    g = fid[name]
    g isa HDF5.Group || return false
    "m" in keys(g) || return false
    "n" in keys(g) || return false
    "colptr" in keys(g) || return false
    "rowval" in keys(g) || return false
    "nzval" in keys(g) || return false
    length(size(g["colptr"])) == 1 || return false
    length(size(g["rowval"])) == 1 || return false
    length(size(g["nzval"])) == 1 || return false
    eltype(g["colptr"]) == eltype(g["rowval"]) || return false
    true
end

"""

Return `true` if `h5open(filename)[name]` is a valid `H5SparseMatrixCSC` dataset, and `false` 
otherwise.
"""
function h5isvalidcsc(filename::AbstractString, name::AbstractString)
    HDF5.ishdf5(filename) || return false
    h5open(filename) do fid
        return h5isvalidcsc(fid, name)
    end
end

end