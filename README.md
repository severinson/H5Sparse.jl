# H5Sparse

This package provides `H5SparseMatrixCSC`, an an out-of-core `AbstractSparseMatrixCSC` backed by a HDF5 dataset stored on disk. The main purpose of this package is to provide support for iteratively constructing large sparse matrices that do not fit in memory and for querying a subset of the columns of that matrix.

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

# Construct from an existing file
A = H5SparseMatrixCSC("foo.h5", "A")
A = H5SparseMatrixCSC(fid, "A")

# Append a SparseMatrixCSC to the right; useful for constructing large matrices in an iterative fashion
D = sprand(10, 5, 0.5)
append!(A, D)       # A is now of size (10, 15)

# Reading the entire matrix from disk
sparse(A)           # SparseMatrixCSC
Matrix(A)           # Matrix

# Querying columns, or blocks of columns, is fast, but quering rows is slow
@time A[:, 1];      # 0.000197 seconds (77 allocations: 3.844 KiB)
@time A[:, 1:10];   # 0.000192 seconds (71 allocations: 4.234 KiB)
@time A[1, :];      # 0.001479 seconds (1.35 k allocations: 69.109 KiB)
```