# Module for evaluating model

module eval_mod

using Plots
using MLBase

export eval_model
export evaluate_performance


global template = """
    Seed: {8}
    Percentage of data for training: {18}
    Percentage of positive instances in the training data: {12}
    Prime: {11}
    Number of epochs: {9}
    Used batchsize: {10}
    Learning rate: {19}
    Recall: {1}
    Precision: {2}
    F-score: {7}
    Confusion matrix: 
                        |   | T | F |
                        | P | {5} | {6} |
                        | N | {3} | {4} |

    Amount of neurons in each part of the network:
        |  BM_path |  BM_query |  PM  |
      1.|     {13}    |     {15}     |  {17}  |
      2.|     {14}    |     {16}     |      |

    Activation functions in parts of the network:
      |  BM_path |  BM_query |  PM  |
    1.|     {20}    |     {22}     |  {24}  |
    2.|     {21}    |     {23}     |      |
    
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

struct Evaluation
    recall::Float64
    precision::Float64
    true_pos::Int64
    false_pos::Int64
    true_neg::Int64
    false_neg::Int64
    F_score::Float64
end

# Evaluate the performance of trained model
function evaluate_performance(true_labels::Vector{Int64}, predicted_labels::Vector{Int64})::Evaluation

    # Rate of correct predictions
    #corr_rate = correctrate(true_labels, predicted_labels)   # Float64

    # Create consufion matrix
    conf_matrix = confusmat(2, true_labels.+1, predicted_labels)

    # Compute ROC curve, output is an Array{ROCNums{Int}}(undef, nt)
    #roc_curve = roc(true_labels, predicted_labels)

    # Calculate precison 
    precis = (conf_matrix[2,2])/(conf_matrix[2,2] + conf_matrix[1,2])

    # Calculate recall
    recal = (conf_matrix[2,2])/(conf_matrix[2,2] + conf_matrix[2,1])

    # F_score
    f_score = round(2 * (precis * recal)/(precis+recal), digits = 2)

    return Evaluation(round(recal, digits = 2), round(precis, digits = 2), conf_matrix[1,1], conf_matrix[2,1], conf_matrix[2,2], conf_matrix[1,2], f_score)
    #return tuple(round(corr_rate, digits = 2), round(precis, digits = 2), conf_matrix[2,2], conf_matrix[1,2], conf_matrix[1,1], conf_matrix[2,1])
end

# Evaluation of the model
function eval_model(filename::String, true_labels::Vector{Int64}, predicted_labels::Vector{Int64}, param_of_network)::Nothing

    # Performance 
    permormance_val = evaluate_performance(true_labels, predicted_labels)
    
    # Combine structs for file  writing
    new_tuple = combine_structs_to_tuple(permormance_val, param_of_network)

    # Write performance
    append_eval(filename, new_tuple)
    return nothing
end

# Merging of structs into a single tuple
function struct_to_tuple(s)::Tuple
    field_values = Vector{Any}(undef, fieldcount(typeof(s)))
    for (i, field_name) in enumerate(fieldnames(typeof(s)))
       field_values[i] = getfield(s, field_name)
    end
    return tuple(field_values...)
end

function combine_structs_to_tuple(s1, s2)::Tuple
    tuple1 = struct_to_tuple(s1)
    tuple2 = struct_to_tuple(s2)
    return (tuple1..., tuple2...)
end

function plot_acc_vs(acc::Vector{Float64}, vss::Vector{Int64})::Nothing
    #Plots.plot(vss, acc)

    scatter(vss, acc, label="Data Points", xticks=(vss, string.(vss)))  # Convert x-values to strings for labels
    savefig("acc_vs.png")
    return nothing
end






end