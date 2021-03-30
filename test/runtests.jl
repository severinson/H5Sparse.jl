using H5Sparse
using Test
using SparseArrays

@testset "H5Sparse.jl" begin
    B = sprand(10, 10, 0.5)
    filename = tempname()
    name = "A"
    A = H5SparseMatrixCSC(filename, name, B)    
    @test size(A) == size(B)
    @test eltype(A) == eltype(B)

    # getitem
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
    @test Matrix(A) == Matrix(B)

    # appending
    C = sprand(10, 5, 0.5)
    append!(A, C)
    @test sparse(A) ≈ hcat(B, C)
    D = sprand(10, 11, 0.1)
    append!(A, D)
    @test sparse(A) ≈ hcat(B, C, D)
end
