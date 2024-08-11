using Revise
using  HTTP, Random, IterTools, Distributed, JLD2
#addprocs(2)


@everywhere begin
    using Mill, OneHotArrays, Flux, Statistics, DataFrames, CSV, SparseArrays
    #file_lock = ReentrantLock()
end

function main(ranges::Vector{Vector{Float64}}, amount_of_urls_to_use::Int64, percentage_of_data::Float64=0.6, prime::Int64=2053, ngrams::Int64=3, base::Int64=256)::Nothing


    # Generate random number every time
    random_seed = rand(UInt32)
    #println("Seed: ", random_seed)
    new_indexing = randcycle(MersenneTwister(random_seed), amount_of_urls_to_use)     # Generate permutation of set length, seed the generator with random_seed
    amount_of_training_data = Int(round(percentage_of_data*length(new_indexing)))     # Specify the amount of data wanted
    split_indx = Int(round(0.8*length(new_indexing)))   # Split index for selecting validation set from evaluation set


    # Read lines
    all_urls = JLD2.load("final_mix_urls.jld2", "all_urls")[new_indexing]    #  Load URL addresses
    all_labels = JLD2.load("final_mix_labels.jld2", "all_labels")[new_indexing]    # Load labels for URLs

    println(sum(all_labels)/lenght(all_labels))
    println(a)
    # Create the training data and the evaluation data 
    training_lines = all_urls[new_indexing[1:amount_of_training_data]]      # Choose training lines
    training_labels = all_labels[new_indexing[1:amount_of_training_data]]     # Choose training labels
    

    eval_lines = all_urls[new_indexing[amount_of_training_data+1:split_indx]]      # Choose eval lines
    eval_labels = all_labels[new_indexing[amount_of_training_data+1:split_indx]]     # Choose eval labels

    valid_lines = all_urls[new_indexing[split_indx+1:end]]      # Choose valid lines
    valid_labels = all_labels[new_indexing[split_indx+1:end]]     # Choose valid labels

    # Info about data
    # println("Number of positive training instances: ", sum(training_labels))
    # println("Nubmer of negative training instances: ", length(training_labels) - sum(training_labels))
    # println("Number of positive eval instances: ", sum(eval_labels))
    # println("Number of negative eval instances: ", length(eval_labels) - sum(eval_labels))
  
    # Prepare training and evaluation data
    training_urls = url_to_mill.(training_lines, ngrams, prime, base)
    eval_urls = url_to_mill.(eval_lines, ngrams, prime, base)
    valid_urls = url_to_mill.(valid_lines, ngrams, prime, base)
    
    # Construct additional parameters
    additional_params = Additional_params(Int64(random_seed), prime, round(100/percentage_of_data))

    # Construct parameter combinations to search
    par_comb = filter_grid(par_grid(ranges))     # Create a grid with possible values for each hyperparameter
    training_params = Training_params.(par_comb)    # Convert ranges into structs

    # Grid search for hyperparameters
    grid_search(training_urls, eval_urls, valid_urls, training_labels, eval_labels, valid_labels, training_params, additional_params)

    return nothing
end

# Struct that holds training parameters for grid search
@everywhere mutable struct Training_params{T<:Int64, S<:Float64}
    epochs::T
    batchsize::T
    domain_neur_1::T
    domain_neur_2::T
    path_neur_1::T
    path_neur_2::T
    query_neur_1::T
    query_neur_2::T
    path_query_neur::T
    learning_rate::S
    treshold::S
end
# Outer constructor
@everywhere function Training_params(vec::Vector{Float64})
    if length(vec) != 11
        error("Input vector must have exactly 8 elements")
    end
    ints = Int64.(vec[1:end-2])
    floats = vec[end-1:end]
    Training_params(ints..., floats...)
end
# Structure that holds additional parameters for grid search
@everywhere struct Additional_params{T<:Int64, S<:Float64}
    seed::T
    prime::T
    percentage_of_data::S
end

# Takes in a URL as string and divides it, then creates bagnodes and finally a single product node for single URL
function url_to_mill(input::String, ngrams::Int64, prime::Int64, base::Int64)::ProductNode
    url = HTTP.URI(input) # Create "URL" type
    
    # Create parts of the URL
    #host = url.host
    path = url.path
    query = url.query

    # Prepare input into functions
    #host = String.(split(host,"."))
    path = filter(!isempty, String.(split(path,"/")[2:end]))
    path = isempty(path) ? [""] : path
    query = String.(split(query,"&"))

    # Construct the structure of the model node
    node = Mill.ProductNode((#transform_url(host,ngrams, prime, base),
                             transform_url(path, ngrams, prime, base), 
                             transform_url(query, ngrams, prime, base)
                                ))
    return node
end

# Transforms vector string[URL] into a BagNode
function transform_url(input::Vector{String}, ngrams::Int64, prime::Int64, base::Int64, sparse = false)::BagNode

    matrix = Mill.NGramMatrix(input, ngrams, base, prime)    # Create NGramMatrix from the string
    #matrix = sparse ? SparseMatrixCSC(matrix) : matrix
    bn = Mill.BagNode(Mill.ArrayNode(matrix), [1:length(input)])     # NgramMatrix can act as ArrayNode data
    return bn
end


# Struct for holding evaluation results of a model
@everywhere struct Evaluation{T<:Int64, S<:Float64}
    recall::S
    precision::S
    true_pos::T
    false_pos::T
    true_neg::T
    false_neg::T
    F_score::S
    Mathew::S
end

# Count the confusion matrix from the predicted and true labels
@everywhere function confusion_matrix(true_labels::Vector{Int64}, predicted_labels::Vector{Int64})::Matrix{Int64}
    # Layout:
    #                [TP, FP]
    #                [FN, TN]
    matrix = zeros(Int64, 2, 2)
    # Iteratation
    for (true_label, pred_label) in zip(true_labels, predicted_labels)
        if true_label == 1 && pred_label == 1
            matrix[1, 1] += 1  # True Positive
        elseif true_label == 1 && pred_label == 0
            matrix[2, 1] += 1  # False Negative
        elseif true_label == 0 && pred_label == 1
            matrix[1, 2] += 1  # False Positive
        else
            matrix[2, 2] += 1  # True Negative
        end
    end
    # TP = sum((true_labels .== 1) .& (predicted .== 1))
    # TN = sum((true_labels .== 0) .& (predicted .== 0))
    # FP = sum((true_labels .== 0) .& (predicted .== 1))
    # FN = sum((true_labels .== 1) .& (predicted .== 0))
    
    return matrix
end

# Phi coeficient ~ mathews correlation coefficient
@everywhere function MCC(C::Matrix{Int64})::Float64
    #                [TP, FP]
    #                [FN, TN]
    return (C[1,1]*C[2,2] - C[1,2]*C[2,1])/(sqrt((C[1,1]+C[1,2])*(C[1,1]+C[2,1])*(C[2,2]+C[2,1])*(C[2,2]+C[1,2])))
end

# Evaluate the performance of trained model
@everywhere function evaluate_performance(true_labels::Vector{Int64}, predicted_labels::Vector{Int64})::Evaluation
    #         [TP, FP]
    #         [FN, TN]

    # Create consufion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate precison 
    precis = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[1,2])

    # Calculate recall
    recall = (conf_matrix[1,1])/(conf_matrix[1,1] + conf_matrix[2,1])

    # F_score
    f_score = round(2 * (precis * recall)/(precis+recall), digits = 2)

    mathew = MCC(conf_matrix)
    println("Confusion matrix: ", conf_matrix)
    println("Mathews correlation coefficient: ", MCC(conf_matrix))

    return Evaluation(round(recall, digits = 3), round(precis, digits = 3), conf_matrix[1,1], conf_matrix[1,2], conf_matrix[2,2], conf_matrix[2,1], f_score, round(mathew, digits=3))
end

# Create the model with specific parameters
@everywhere function create_model(param::Training_params, prime::Int64)::ProductModel                          

    mod = ProductModel(tuple(
        #BagModel(ArrayModel(Flux.Chain( Dense(prime=>prime, gelu), Dropout(0.6), Dense(prime=>param.domain_neur_1), BatchNorm(param.domain_neur_1), x->gelu(x), Dense(param.domain_neur_1=>param.domain_neur_2, gelu), BatchNorm(param.domain_neur_2))),
         #   AggregationStack(SegmentedMean(param.domain_neur_2), SegmentedSum(param.domain_neur_2))),

        BagModel(ArrayModel(Flux.Chain( Dense(prime=>prime, gelu), Dropout(0.5), Dense(prime=>param.path_neur_1), BatchNorm(param.path_neur_1), x->gelu(x), Dense(param.path_neur_1=>param.path_neur_2, gelu), BatchNorm(param.path_neur_2))),
            AggregationStack(SegmentedMean(param.path_neur_2), SegmentedMax(param.path_neur_2))),

        BagModel(ArrayModel(Flux.Chain( Dense(prime=>prime, gelu), Dropout(0.5), Dense(prime=>param.query_neur_1), BatchNorm(param.query_neur_1), x->gelu(x), Dense(param.query_neur_1=>param.query_neur_2, gelu), BatchNorm(param.query_neur_2))),
            AggregationStack(SegmentedMean(param.query_neur_2), SegmentedMax(param.query_neur_2))),
            
        ),
        Flux.Chain(BatchNorm(2*(param.path_neur_2+param.query_neur_2)), Dense(2*(param.domain_neur_2+param.path_neur_2+param.query_neur_2)=>2*(param.domain_neur_2+param.path_neur_2+param.query_neur_2), gelu), Dropout(0.45),
        Dense(2*(param.domain_neur_2+param.path_neur_2+param.query_neur_2)=>param.path_query_neur), BatchNorm(param.path_query_neur), x->gelu(x),  Dense(param.path_query_neur=>2, gelu)
        ))

    return mod
end

# Training function
@everywhere function train(training_urls::Vector{<:ProductNode}, eval_urls::Vector{<:ProductNode}, valid_urls::Vector{<:ProductNode}, training_labels::Vector{Int64}, valid_labels::Vector{Int64}, param::Training_params{Int64}, prime::Int64)::Tuple 

    # Create model of the network
    model = create_model(param, prime) # What with the activation functions
    
    # Create a loss
    loss(x, y) = Flux.Losses.logitbinarycrossentropy(x, OneHotArrays.onehotbatch(y .+ 1, 1:2))  # Onehot inside from training labels

    # Create minibatches
    training_data_loader = Flux.DataLoader((training_urls, training_labels), batchsize = param.batchsize, shuffle = true, partial = false)
    valid_data_loader = Flux.DataLoader((valid_urls, valid_labels), batchsize = param.batchsize, shuffle = true, partial = false)


    # Early stopping
    # Stopping  criteria for training loop
    state = Dict("best_loss" => Inf, "no_improv" => 0, "patience"=> 3)

    opt = Flux.Optimiser(Adam(param.learning_rate))
    opt_state = Flux.setup(opt, model)

    #println(a)

    # Early stopping callback
    function early_stop_cb()::Bool
        eval_loss = mean(loss(model(batch[1]), batch[2]) for batch in valid_data_loader)
        println(eval_loss)
        println(state["no_improv"])
        if eval_loss < state["best_loss"]
            state["best_loss"] = eval_loss
            state["no_improv"] = 0
        else
            state["no_improv"] += 1
        end
        return state["no_improv"] > state["patience"]
    end

    stopped_at = false  # Early stop signalizer

    trainmode!(model)
    # Training loop
    for i in 1:param.epochs
        @info "Epoch number $i."
        is_finite = true
        # Batch work
        for (x_batch, y_batch) in training_data_loader

            # Calculate the loss and gradients
            val, grads = Flux.withgradient(model) do m
                result = m(x_batch)
                loss(result, y_batch)
            end
            
            #push!(losses, val)  # Save losses

            # Check for valid loss value
            if !isfinite(val)
                @warn "Loss is $val." 
                is_finite = false # Remark the loss
                break
    
            end
           
            #println(sum(model.ms[3].im.m.layers[3].weight))
            # Update the parameters of the model (grads[1] is a nested set of NamedTuples)
            Flux.update!(opt_state, model, grads[1])
        end
         # Early stopping
         if early_stop_cb()
            stopped_at = true
            break
        end
        # Control finite loss
        if !is_finite
            println("Loss is not finite at epoch $i.")
            break
        end
    end

    testmode!(model) # Turn off dropout

    # println("Permormace on the training set:")
    # train_probs = softmax(model(training_urls))[1,1:end]
    # training_predicted = cold_treshold(train_probs, param.treshold)
    # training_metric = evaluate_performance(training_labels, training_predicted)
    # println("----------------")

    #println(softmax(model(eval_urls[1:50])))
    probs = softmax(model(eval_urls))[1,1:end]   # Transform output into probabilities

    # Return probabilities and trained model and if early early stopping occured
    return probs, model, stopped_at
end

# Cold treshold for converting a vector of probabilites into predictions
@everywhere function cold_treshold(probs::Vector{Float32}, treshold = 0.5)
    return map(x-> x <= treshold ? 1 : 0, probs)
end

# Create a channel for asyn. writing
@everywhere channel = RemoteChannel(()->Channel{DataFrame}(10))

# Logger that controls takes results from the channel and call the append function to write into a CSV file
function result_logger(channel)
    println("Logger active")
    try 
        while isopen(channel) || isready(channel) # Condition on running the writing from channel
            try
                if isready(channel)
                    data = take!(channel);
                    println("Data received.");  
                    append_data_to_csv(data);
                else
                    sleep(0.1)
                end
            catch e
                println("Error processing data: $e")  # Error handling
            end
        end
    catch e
        println("Error in logger: $e")  # Error handling
    finally
        println("Logger closed")
    end
end

# Append data into a CSV file 
function append_data_to_csv(data)
    filename = "results_par.csv"
    #df = DataFrame(data, column_names)
    if isfile(filename)
        CSV.write(filename, data, append= true)
    else
        CSV.write(filename, data)
    end
end

# Performs grid search and writes eval data into a csv file
@everywhere function grid_search(training_urls::Vector{<:ProductNode}, eval_urls::Vector{<:ProductNode}, valid_urls::Vector{<:ProductNode}, training_labels::Vector{Int64}, eval_labels::Vector{Int64}, valid_labels::Vector{Int64}, training_param::Vector{Training_params{Int64, Float64}}, additional_param::Additional_params)::Nothing
    @async result_logger(channel)
    # Cycle for searching the grid
    @sync @distributed for i in 1:length(training_param)
        probs, model, stopped = train(training_urls, eval_urls, valid_urls, training_labels, valid_labels, training_param[i], additional_param.prime)   # train the network
        predicted_labels = cold_treshold(probs, training_param[i].treshold)     # Convert probabilities to labels based on treshold
        evaluation_metric = evaluate_performance(eval_labels, predicted_labels)    # Evaluate  performance with current network

        # Construct data to be logged
        data = DataFrame(Epochs = ["$(training_param[i].epochs)"], 
        Stopped = ["$(stopped)"],
        Batchsize = ["$(training_param[i].batchsize)"],
       DN1 = ["$(training_param[i].domain_neur_1)"],
       DN2 = ["$(training_param[i].domain_neur_2)"],
       PN1 = ["$(training_param[i].path_neur_1)"],
       PN2 = ["$(training_param[i].path_neur_2)"],
       QN1 = ["$(training_param[i].query_neur_1)"],
       QN2 = ["$(training_param[i].query_neur_2)"],
       DQP = ["$(training_param[i].path_query_neur)"],
       Learning_rate = ["$(training_param[i].learning_rate)"],
       Treshold =["$(training_param[i].treshold)"],
       Precision = ["$(evaluation_metric.precision)"],
       Recall = ["$(evaluation_metric.recall)"],
       F_score = ["$(evaluation_metric.F_score)"],
       Mathews = ["$(evaluation_metric.Mathew)"],
       TP = ["$(evaluation_metric.true_pos)"],
       FP = ["$(evaluation_metric.false_pos)"],
       FN = ["$(evaluation_metric.false_neg)"],
       TN =["$(evaluation_metric.true_neg)"])

        if evaluation_metric.Mathew >= 0.8
            # Save model
            filename = "$(training_param[i].epochs)_$(training_param[i].batchsize)_$(training_param[i].learning_rate)_$(training_param[i].domain_neur_1)_$(training_param[i].domain_neur_2)_$(training_param[i].path_neur_1)_$(training_param[i].path_neur_2)_$(training_param[i].query_neur_1)_$(training_param[i].query_neur_2)_$(training_param[i].path_query_neur)"   # Construct filename
            println("Good model")
            #save("files/$filename.jld2", "Evaluation_metrics", evaluation_metric, "Training_params", training_param[i],"Additional_params", additional_param, "Model", model)   # Save the current network and its parameters                  
        end
        put!(channel, data) # Put the data into a channel
    end

    @sync close(channel)

    return nothing
end


# Custom structure for powers of two
struct POT{T<:Int} <: AbstractVector{Int}
    first::T
    last::T
end
Base.size(p::POT) = (Int(log2(p.last/p.first)+1),)
Base.getindex(p::POT, i::Int) = 2^(log2(p.first) + (i - 1))
function ct(p::POT)::Vector{Int64}
    return [getindex(p,i) for i in 1:length(p)]
end

# Create vector of vectors of unique combinations of parameters
function par_grid(X::Vector{Vector{Float64}})::Vector{Vector{Float64}}
    prod = IterTools.product(X...)
    return unique([collect(t) for t in prod])
end

# Filter grid to only include combinations of parameters that are valid
function filter_grid(X::Vector{Vector{Float64}})::Vector{Vector{Float64}}
    Y = Vector{Vector{Float64}}()
    for vec in X
        if isequal(vec[3], vec[5]) && isequal(vec[3], vec[7]) && isequal(vec[5], vec[7]) 
            if !isequal(vec[3], vec[4]) && !isequal(vec[5], vec[6]) && !isequal(vec[7], vec[8])
                if vec[8]*3 > vec[9]
                    push!(Y, vec)           
                else
                    continue
                end
            else
                continue
            end
        else
            continue
        end
    end
    return Y
end

# epochs, batchsize, domain_neur_1, domain_neur_2, path_neur_1, path_neur_2, query_neur_1, query_neur_2, path_query_neur, learning_rate
#ranges = [[15],  ct(POT(32,512)), ct(POT(1024,2048)), ct(POT(512,1024)), ct(POT(1024,2048)),  ct(POT(512,1024)), ct(POT(1024,2048)), ct(POT(512,1024)), ct(POT(1024,2048)), [0.001,0.005, 0.01]]

ranges = [[15], ct(POT(8,256)), ct(POT(1024,2048)), ct(POT(512,1024)), ct(POT(1024,2048)),  ct(POT(512,1024)), ct(POT(1024,2048)), ct(POT(512,1024)), ct(POT(1024,2048)), [0.001],[0.5,0.9]]
main(ranges, 250000)