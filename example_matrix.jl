using LinearAlgebra
using SymbolicRegression
# using Array{Array{Array{Int64, 2}, 1}, 1}

# X data is a list of features, where each feature is a scalar, vector, or matrix
# Y data is a list of targets, where each target is a scalar, vector, or matrix
X_is = Array{Array{Matrix{Int64},1},1}([[
    Matrix{Int64}([8 6; 6 4]), Matrix{Int64}([7 9; 4 3])
]])
Y_is = Array{Array{Matrix{Int64},1},1}([
    [
        Matrix{Int64}(
            [
                8 6 8 6 8 6
                6 4 6 4 6 4
                6 8 6 8 6 8
                4 6 4 6 4 6
                8 6 8 6 8 6
                6 4 6 4 6 4
            ]
        ),
    ],
    [
        Matrix{Int64}(
            [
                7 9 7 9 7 9
                4 3 4 3 4 3
                9 7 9 7 9 7
                3 4 3 4 3 4
                7 9 7 9 7 9
                4 3 4 3 4 3
            ]
        ),
    ],
],)
X_os = Array{Array{Matrix{Int64},1},1}([[Matrix{Int64}([3 2; 7 8])]])
Y_os = Array{Array{Matrix{Int64},1},1}([[
    Matrix{Int64}(
        [
            3 2 3 2 3 2
            7 8 7 8 7 8
            2 3 2 3 2 3
            8 7 8 7 8 7
            3 2 3 2 3 2
            7 8 7 8 7 8
        ]
    ),
]])

println(size(X_is), size(Y_is), size(X_os), size(Y_os))

# The python solution is to find the equation that generates the Y data from the X data
# X2 = np.tile(X,3)
# X3 = X2[:,::-1]
# Y_pred = np.vstack([X2, X3, X2])

# Julia solution
# function generate_Y_pred()
#     println("Generate Y_pred")
# end

# Function to generate Y data from X data
function generate_Y_pred(X::Matrix{T})::Matrix{T} where T
    # Tile the matrix 3 times along the columns
    X2 = repeat(X, 1, 3)

    # Flatten each row of X2 into a single row
    X2_flat = map(row -> reduce(vcat, row), eachrow(X2))
    # Alternative: X2_flat = [reduce(vcat, row) for row in eachrow(X2)]

    # Create X3 by reversing each row of X2_flat
    # X3 = [row[end:-1:1] for row in X2_flat]
    X3 = map(reverse, X2_flat)

    # Vertically concatenate rows to form Y_pred
    Y_pred = vcat(X2_flat, X3, X2_flat)

    return Matrix{T}(stack(Y_pred, dims=1))
end

z = generate_Y_pred(X_is[1][1])
println(1, ":", z == Y_is[1][1], typeof(z), typeof(Y_is[1][1]))

Y_pred_example = generate_Y_pred(X_os[1][1])
println("Predicted Y from the example X:", Y_pred_example == Y_os[1][1])
println(Y_pred_example)

# Operators:
# unary_operators=[repeat, reverse, vcat, eachrow]
# binary_operators=[vcat, map, reduce]

options = SymbolicRegression.Options(;
    binary_operators=[vcat, map, reduce],
    unary_operators=[repeat, reverse, vcat, eachrow, stack],
    populations=20,
)

# Seach
hall_of_fame = equation_search(
    X_is,
    Y_is;
    niterations=40,
    options=options,
    # parallelism=:multithreading
)

dominating = calculate_pareto_frontier(hall_of_fame[1])

trees = [member.tree for member in dominating]

tree = trees[end]
output, did_succeed = eval_tree_array(tree, X, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
