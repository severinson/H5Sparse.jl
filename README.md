# H5Sparse.jl

This package provides `H5SparseMatrixCSC`, an an out-of-core `AbstractSparseMatrixCSC` backed by a dataset stored on disk of type `<:HDF5.H5DataStore`, e.g., a `HDF5.File`; see the [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) documentation. 

The intended workflow is:

1. Construct a large sparse matrix that does not fit in memory by repeatedly concatenating a `H5SparseMatrixCSC` with matrices of type `SparseMatrixCSC`, which may be generated one at a time and then discarded to free up memory. Each concatenation writes the columns of the `SparseMatrixCSC` to the file backing the `H5SparseMatrixCSC`.
2. Read a subset of the columns of the resulting `H5SparseMatrixCSC` into memory at a time for processing.

Since Julia matrices are stored in column-major format, for efficiency only horizontal concatenation (`hcat`) is supported. By default, the backing file is compressed with [blosc](https://www.blosc.org/), resulting in exceptionally small files, and making reading from disk very fast.

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

# Load a H5SparseMatrixCSC matrix into memory using SparseArrays.sparse
# Only the columns covered by this particular H5SparseMatrixCSC will be read from disk
# However, A must cover all rows of the underlying matrix
sparse(A)           # SparseMatrixCSC
sparse(A[:, 1:4])   # The first 4 columns of A as a new SparseMatrixCSC
sparse(A[1:4, :])   # Results in an error (not implemented)

# If converting to a dense matrix, Matrix(sparse(A)) is likely orders of magnitude faster than calling Matrix(A) directly
# (since Matrix(A) does not take advantage of A being sparse)
Matrix(A)           # Matrix (slow)
Matrix(sparse(A))   # Matrix (fast)
```