using H5Sparse
using Test
using HDF5, SparseArrays

@testset "H5Sparse.jl" begin
    B = sprand(10, 10, 0.5)
    filename = tempname()
    name = "A"

    # constructors
    A = H5SparseMatrixCSC(filename, name, B)    
    @test size(A) == size(B)
    @test size(A, 1) == size(B, 1)
    @test size(A, 2) == size(B, 2)
    @test eltype(A) == eltype(B)
    @test A == B

    A = H5SparseMatrixCSC(filename, name)
    @test A == B
    
    fid = h5open(filename, "cw")
    A = H5SparseMatrixCSC(fid, name)
    @test A == B

    A = H5SparseMatrixCSC(fid, name, 1:size(B, 1), 1:size(B, 2))
    @test sparse(A) ≈ B

    A = H5SparseMatrixCSC(fid, name, :, :)
    @test sparse(A) ≈ B

    A = H5SparseMatrixCSC(fid, name, :, 3:6)
    @test sparse(A) ≈ B[:, 3:6]

    # getitem
    A = H5SparseMatrixCSC(filename, name)    
    for col in 1:size(B, 2)
        @test A[:, col] == B[:, col]
    end
    for row in 1:size(B, 1)
        @test A[row, :] == B[row, :]
    end
    for row in 1:size(B, 1)
        for col in 1:size(B, 2)
            @test A[row, col] ≈ B[row, col]
        end
    end
    for col in 1:size(B, 2)
        @test A[:, 1:col] ≈ B[:, 1:col]
    end

    # conversion
    @test sparse(A) == B
    @test sparse(A[:, 1:size(A, 2)]) == B
    @test sparse(A[:, 2:size(A, 2)-1]) == B[:, 2:size(B, 2)-1]
    @test Matrix(A) == Matrix(B)

    # concatenation
    C = sprand(10, 5, 0.5)
    A = hcat(A, C)
    @test sparse(A) ≈ hcat(B, C)
    D = sprand(10, 11, 0.1)
    A = hcat(A, D)
    @test sparse(A) ≈ hcat(B, C, D)
    A = hcat(A, C, D)
    @test sparse(A) ≈ hcat(B, C, D, C, D)

    # h5isvalidcsc
    @test H5Sparse.h5isvalidcsc(fid, name)
    @test H5Sparse.h5isvalidcsc(filename, name)
    delete_object(fid[name], "m")
    @test !H5Sparse.h5isvalidcsc(fid, name)
    fid["foo"] = 10
    @test !H5Sparse.h5isvalidcsc(fid, "foo")
end
