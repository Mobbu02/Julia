# Module for evaluating model

module eval_mod

#using Plots
using MLBase

export gather_decisions
export evaluate_model


global template = """
    Seed: {3}
    Correct rate of predictions: {1}
    Precision = {2}
    Confusion matrix: 
                        |   | T | F |
                        | P | {4} | {5} |
                        | N | {6} | {7} |

    Amount of neurons in each part of the network:
        |  BM_path |  BM_query |  PM  |
      1.|     {8}    |     {9}     |  {12}  |
      2.|     {10}    |      {11}    |      |
"""

# Append evaluation of model at the end of the Model_eval.txt file
function append_eval(filename::String, values::Tuple)::Nothing
    filled_temp = template
    for(i, val) in enumerate(values)
        pl_hl = "{$i}"
        filled_temp = replace(filled_temp, pl_hl => string(val))
    end
    open(filename, "a") do io
        write(io, filled_temp * "\n" * "\n")
    end
    return nothing
end

# Takes in a matrix from the model of size 2xnumber. Each column = probabilities of instance beinging negative or positive
function gather_decisions(X::Matrix{Float32})::Vector{Int64}
    predicted_labels = zeros(Int64, size(X,2))

    for i in 1:size(X,2)
        predicted_labels[i] = X[1,i] > 0.5 ? 1 : 0
    end
    return predicted_labels
end

# Evaluate the performance of trained model
function evaluate_performance(X::Matrix{Float32}, true_labels::Vector{Int64}, seed::Int64)::Tuple
    # Get predicted labels
    predicted_labels = gather_decisions(X) .+ 1

    # Rate of correct predictions
    corr_rate = correctrate(true_labels, predicted_labels)   # Float64

    # Create consufion matrix
    conf_matrix = confusmat(2, true_labels, predicted_labels)

    # Compute ROC curve, output is an Array{ROCNums{Int}}(undef, nt)
    #roc_curve = roc(true_labels, predicted_labels)

    # Calculate precison 
    #precis = precision(roc_curve)  #Float64
    precis = 1

    return tuple(round(corr_rate, digits = 2), round(precis, digits = 2), seed, conf_matrix[2,2], conf_matrix[1,2], conf_matrix[1,1], conf_matrix[2,1])
end


function eval_model(filename::String, M::Matrix{Float32}, true_labels::Vector{Int64}, seed::Int64, neurons::Tuple)::Nothing

    # Performance 
    permormance_val = evaluate_performance(M, true_labels .+1, seed)
    new_tuple = (permormance_val..., neurons...)
    
    # Write performance
    append_eval(filename, new_tuple)
    return nothing
end

function plot(x, y, iteration)

end





end