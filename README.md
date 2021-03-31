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