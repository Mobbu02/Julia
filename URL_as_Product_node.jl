using Revise
using Mill, HTTP, JLD2, Random, OneHotArrays, Flux, IterTools, DataFrames, CSV, SparseArrays, Plots, Statistics, Profile, Plots


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

    # Training data
    training_subset = new_indexing[1:amount_of_training_data]
    training_lines = all_urls[training_subset]      # Choose training lines
    training_labels = all_labels[training_subset]     # Choose training labels
    
    # Evaluation data
    eval_subset = new_indexing[amount_of_training_data+1:split_indx]
    eval_lines = all_urls[eval_subset]      # Choose eval lines
    eval_labels = all_labels[eval_subset]     # Choose eval labels

    # Check for "goodness" of the model
    length_of_eval = length(eval_labels)
    mal = sum(eval_labels.==1)/length_of_eval
    println("Percentage of malicious URLs within eval data: ", mal)


    # Validation data
    validation_subset = new_indexing[split_indx+1:end]
    valid_lines = all_urls[validation_subset]      # Choose valid lines
    valid_labels = all_labels[validation_subset]     # Choose valid labels
  
    # Prepare training and evaluation and validation data
    training_urls = url_to_mill.(training_lines, ngrams, prime, base)
    eval_urls = url_to_mill.(eval_lines, ngrams, prime, base)
    valid_urls = url_to_mill.(valid_lines, ngrams, prime, base)
    
    # Construct additional parameters
    additional_params = Additional_params(Int64(random_seed), prime, round(100/percentage_of_data))

    # Construct parameter combinations to search
    par_comb = par_grid(ranges)     # Create a grid with possible values for each hyperparameter
    training_params = Training_params.(par_comb)    # Convert ranges into structs

    # Grid search for hyperparameters
    grid_search(training_urls, eval_urls, valid_urls, training_labels, eval_labels, valid_labels, training_params, additional_params)

    return nothing
end

# Struct that holds training parameters for grid search
mutable struct Training_params{T<:Int64, S<:Float64}
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
function Training_params(vec::Vector{Float64})
    if length(vec) != 11
        error("Input vector must have exactly 11 elements")
    end
    ints = Int64.(vec[1:end-2])
    floats = vec[end-1:end]
    Training_params(ints..., floats...)
end

# Structure that holds additional parameters for grid search
struct Additional_params{T<:Int64, S<:Float64}
    seed::T
    prime::T
    percentage_of_data::S
end

# Takes in a URL as string and divides it, then creates bagnodes and finally a single product node for single URL
function url_to_mill(input::String, ngrams::Int64, prime::Int64, base::Int64)::ProductNode
    url = HTTP.URI(input) # Create "URL" type
    
    # Create parts of the URL
    host = url.host
    path = url.path
    query = url.query

    # Prepare input into functions
    host = String.(split(host,"."))
    path = filter(!isempty, String.(split(path,"/")[2:end]))
    path = isempty(path) ? [""] : path
    query = String.(split(query,"&"))

    # Construct the structure of the model node
    node = Mill.ProductNode((transform_url(host,ngrams, prime, base),
                             transform_url(path, ngrams, prime, base), 
                             transform_url(query, ngrams, prime, base)
                                ))
    return node
end

# Transforms vector string[URL] into a BagNode
function transform_url(input::Vector{String}, ngrams::Int64, prime::Int64, base::Int64, sparse = false)::BagNode
    # # Check for empty path
    # if isempty(input)
    #     input = [""]
    # end
    matrix = Mill.NGramMatrix(input, ngrams, base, prime)    # Create NGramMatrix from the string
    #matrix = sparse ? SparseMatrixCSC(matrix) : matrix
    bn = Mill.BagNode(Mill.ArrayNode(matrix), [1:length(input)])     # NgramMatrix can act as ArrayNode data
    return bn
end

# Struct for holding evaluation results of a model
struct Evaluation{T<:Int64, S<:Float64}
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
function confusion_matrix(true_labels::Vector{Int64}, predicted_labels::Vector{Int64})::Matrix{Int64}
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
function MCC(C::Matrix{Int64})::Float64
    #                [TP, FP]
    #                [FN, TN]
    return (C[1,1]*C[2,2] - C[1,2]*C[2,1])/(sqrt((C[1,1]+C[1,2])*(C[1,1]+C[2,1])*(C[2,2]+C[2,1])*(C[2,2]+C[1,2])))
end

# Evaluate the performance of trained model
function evaluate_performance(true_labels::Vector{Int64}, predicted_labels::Vector{Int64})::Evaluation
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
    println("Precision:", precis)
    println("Recall: ", recall)
    println("Mathews correlation coefficient: ", MCC(conf_matrix))

    return Evaluation(round(recall, digits = 3), round(precis, digits = 3), conf_matrix[1,1], conf_matrix[1,2], conf_matrix[2,2], conf_matrix[2,1], f_score, round(mathew, digits=3))
end

# Create the model with specific parameters
function create_model(param::Training_params, prime::Int64)::ProductModel                          

    # mod = ProductModel(tuple(
    #     BagModel(ArrayModel(Flux.Chain(Dense(prime=>param.domain_neur_1, bias = true), BatchNorm(param.domain_neur_1), x->rrelu(x), Dense(param.domain_neur_1=>param.domain_neur_2, bias = true, rrelu), Dropout(0.6,active=true), Dense(param.domain_neur_2=>Int(param.domain_neur_2/2), rrelu), Dropout(0.5,active=true))), 
    #             AggregationStack(SegmentedMean(Int(param.domain_neur_2/2)), SegmentedLSE(Int(param.domain_neur_2/2))), ),

    #     BagModel(ArrayModel(Flux.Chain(Dense(prime => param.path_neur_1, bias = true), BatchNorm(param.path_neur_1), x->rrelu(x), Dense(param.path_neur_1=> param.path_neur_2, bias=true, rrelu), Dropout(0.6,active=true), Dense(param.path_neur_2=>Int(param.path_neur_2/2), rrelu), Dropout(0.5,active=true) ) ),  #
    #             AggregationStack(SegmentedMean(Int(param.path_neur_2/2)), SegmentedLSE(Int(param.path_neur_2/2))), ),
                        
    #     BagModel(ArrayModel(Flux.Chain(Dense(prime => param.query_neur_1, bias = true), BatchNorm(param.query_neur_1), x->rrelu(x), Dense(param.query_neur_1=> param.query_neur_2, bias=true, rrelu), Dropout(0.6,active=true), Dense(param.query_neur_2=>Int(param.query_neur_2/2), rrelu), Dropout(0.5,active=true))), 
    #             AggregationStack(SegmentedMean(Int(param.query_neur_2/2)), SegmentedLSE(Int(param.query_neur_2/2))), )

    # ), Flux.Chain(Dense((param.domain_neur_2 + param.path_neur_2 + param.query_neur_2) => param.path_query_neur, bias = true),
    #     BatchNorm(param.path_query_neur),
    #     relu,
    #     Dense(param.path_query_neur=>Int(param.path_query_neur/2),bias=true, relu), 
    #     Dropout(0.3, active=true), 
    #     Dense(Int(param.path_query_neur/2)=>Int(param.path_query_neur/4), bias = true, relu), 
    #     Dropout(0.3, active=true),
    #     Dense(Int(param.path_query_neur/4) => 2)))


    # model = ProductModel(tuple(
    #     BagModel(ArrayModel(Flux.Chain(Dense(prime=>param.domain_neur_1, bias = true, gelu), Dropout(0.3, active=true), Dense(param.domain_neur_1=>param.domain_neur_2, bias = true, gelu), Dropout(0.25, active = true), Dense(param.domain_neur_2=>Int(param.domain_neur_2/2), gelu))),
    #             AggregationStack(SegmentedMean(Int(param.domain_neur_2/2)), SegmentedLSE(Int(param.domain_neur_2/2))), ),

    #     BagModel(ArrayModel(Flux.Chain(Dense(prime => param.path_neur_1, bias = true, gelu ), Dropout(0.3, active=true), Dense(param.path_neur_1=> param.path_neur_2, bias=true, gelu), Dropout(0.25, active = true), Dense(param.path_neur_2=>Int(param.path_neur_2/2), gelu) ) ),  #
    #             AggregationStack(SegmentedMean(Int(param.path_neur_2/2)), SegmentedLSE(Int(param.path_neur_2/2))), ),

    #     BagModel(ArrayModel(Flux.Chain(Dense(prime => param.query_neur_1, bias = true, gelu), Dropout(0.3, active=true), Dense(param.query_neur_1=> param.query_neur_2, bias=true, gelu), Dropout(0.25, active = true), Dense(param.query_neur_2=>Int(param.query_neur_2/2), gelu))),
    #             AggregationStack(SegmentedMean(Int(param.query_neur_2/2)), SegmentedLSE(Int(param.query_neur_2/2))), )

    # ), Flux.Chain(Dense((param.domain_neur_2 + param.path_neur_2 + param.query_neur_2) => param.path_query_neur, bias = true, x-> elu(x,2)),
    #     Dropout(0.2, active=true) ,
    #     Dense(param.path_query_neur=>Int(param.path_query_neur/2),bias=true, x-> elu(x,2)),
    #     Dropout(0.2, active=true),
    #     Dense(Int(param.path_query_neur/2)=>Int(param.path_query_neur/4), bias = true, x-> elu(x,2)),
    #     Dense(Int(param.path_query_neur/4)=> 2, x->elu(x,2))))


    mod = ProductModel(tuple(
        BagModel(ArrayModel(Flux.Chain(Dense(prime=>param.domain_neur_1, bias = true, gelu), Dropout(0.3, active=true), Dense(param.domain_neur_1=>param.domain_neur_2, bias = true, gelu), Dropout(0.25, active = true), Dense(param.domain_neur_2=>Int(param.domain_neur_2/2), gelu))),
                AggregationStack(SegmentedMean(Int(param.domain_neur_2/2)), SegmentedLSE(Int(param.domain_neur_2/2))), ),

        BagModel(ArrayModel(Flux.Chain(Dense(prime => param.path_neur_1, bias = true, gelu ), Dropout(0.3, active=true), Dense(param.path_neur_1=> param.path_neur_2, bias=true, gelu), Dropout(0.25, active = true), Dense(param.path_neur_2=>Int(param.path_neur_2/2), gelu) ) ),  #
                AggregationStack(SegmentedMean(Int(param.path_neur_2/2)), SegmentedLSE(Int(param.path_neur_2/2))), ),

        BagModel(ArrayModel(Flux.Chain(Dense(prime => param.query_neur_1, bias = true, gelu), Dropout(0.3, active=true), Dense(param.query_neur_1=> param.query_neur_2, bias=true, gelu), Dropout(0.25, active = true), Dense(param.query_neur_2=>Int(param.query_neur_2/2), gelu))),
                AggregationStack(SegmentedMean(Int(param.query_neur_2/2)), SegmentedLSE(Int(param.query_neur_2/2))), )

    ), Flux.Chain(Dense((param.domain_neur_2 + param.path_neur_2 + param.query_neur_2) => param.path_query_neur, bias = true, x-> elu(x,2)),
        Dropout(0.2, active=true) ,
        Dense(param.path_query_neur=>Int(param.path_query_neur/2),bias=true, x-> elu(x,2)),
        Dropout(0.2, active=true),
        Dense(Int(param.path_query_neur/2)=>Int(param.path_query_neur/4), bias = true, x-> elu(x,2)),
        Dense(Int(param.path_query_neur/4)=> 2, x->elu(x,2))))

    return mod



    printtree(model)

    

    #  mod = ProductModel(tuple(
    #     BagModel(ArrayModel(Flux.Chain(Dense(prime=>param.domain_neur_1, bias = true, relu), Dropout(0.25, active=true), Dense(param.domain_neur_1=>param.domain_neur_1, bias = true, relu), 
    #     Dense(param.domain_neur_1=> param.domain_neur_2, relu), Dropout(0.25, active = true), Dense(param.domain_neur_2=>param.domain_neur_2, relu), Dropout(0.25,active=true),Dense(param.domain_neur_2=>Int(param.domain_neur_2/2), relu))), 
    #             AggregationStack(SegmentedMean(Int(param.domain_neur_2/2)), SegmentedLSE(Int(param.domain_neur_2/2))), ),

    #     BagModel(ArrayModel(Flux.Chain(Dense(prime=>param.path_neur_1, bias = true, relu), Dropout(0.25, active=true), Dense(param.path_neur_1=>param.path_neur_1, bias = true, relu), 
    #     Dense(param.path_neur_1=> param.path_neur_2, relu), Dropout(0.25, active = true), Dense(param.path_neur_2=>param.path_neur_2, relu), Dropout(0.25,active=true),Dense(param.path_neur_2=>Int(param.path_neur_2/2), relu))), 
    #             AggregationStack(SegmentedMean(Int(param.path_neur_2/2)), SegmentedLSE(Int(param.path_neur_2/2))), ),
                        
    #     BagModel(ArrayModel(Flux.Chain(Dense(prime=>param.query_neur_1, bias = true, relu), Dropout(0.25, active=true), Dense(param.query_neur_1=>param.query_neur_1, bias = true, relu), 
    #     Dense(param.query_neur_1=> param.query_neur_2, relu), Dropout(0.25, active = true), Dense(param.query_neur_2=>param.quer1024y_neur_2, relu), Dropout(0.25,active=true),Dense(param.query_neur_2=>Int(param.query_neur_2/2), relu))), 
    #             AggregationStack(SegmentedMean(Int(param.query_neur_2/2)), SegmentedLSE(Int(param.query_neur_2/2))), )

    # ), Flux.Chain(Dense((param.domain_neur_2 + param.path_neur_2 + param.query_neur_2) => param.path_query_neur, bias = true, x-> elu(x,2)),
    #     Dropout(0.1, active=true) ,
    #     Dense(param.path_query_neur=>Int(param.path_query_neur/2),bias=true, x-> elu(x,2)), 
    #     Dropout(0.1, active=true), 
    #     Dense(Int(param.path_query_neur/2)=>Int(param.path_query_neur/4), bias = true, x-> elu(x,2)),
    #     Dropout(0.1, active=true) ,
    #     Dense(Int(param.path_query_neur/4)=> 2, x->elu(x,2))))
    
    # mod = ProductModel(tuple(
    #     BagModel(ArrayModel(Flux.Chain(Dense(2053=>2048, bias = true), BatchNorm(2048), x->gelu(x), Dense(2048=>2048, gelu), Dropout(0.5, active = true),
    #     Dense(2048=>1536), BatchNorm(1536), x->gelu(x), Dense(1536=> 1024, gelu), Dropout(0.5, active=true), Dense(1024=>1024, gelu), BatchNorm(1024), x->gelu(x),
    #     Dense(1024=>512), Dropout(0.5, active=true), Dense(512=>512, gelu), Dropout(0.3,active=true))),

    #    SegmentedMean(512), ),

    #     BagModel(ArrayModel(Flux.Chain(Dense(2053=>2048, bias = true), BatchNorm(2048), x->gelu(x), Dense(2048=>2048, bias = true), Dropout(0.5, active = true),
    #     Dense(2048=>1536), BatchNorm(1536), x->gelu(x), Dense(1536=> 1024), Dropout(0.5, active=true), Dense(1024=>1024), BatchNorm(1024), x->gelu(x),
    #     Dense(1024=>512), Dropout(0.5, active=true), Dense(512=>512, gelu), Dropout(0.3,active=true))),

    #    SegmentedMean(512), ),

    #     BagModel(ArrayModel(Flux.Chain(Dense(2053=>2048, bias = true), BatchNorm(2048), x->gelu(x), Dense(2048=>2048, bias = true), Dropout(0.5, active = true),
    #     Dense(2048=>1536), BatchNorm(1536), x->gelu(x), Dense(1536=> 1024), Dropout(0.5, active=true), Dense(1024=>1024), BatchNorm(1024), x->gelu(x),
    #     Dense(1024=>512), Dropout(0.5, active=true), Dense(512=>512, gelu), Dropout(0.3,active=true))),

    #    SegmentedMean(512), )
    
    # ), Flux.Chain(Dense(1536 => 1024, bias = true), BatchNorm(1024), x->elu(x,2),
    #     Dense(1024=>1024,bias=true, x-> elu(x,2)),
    #     Dropout(0.4, active=true),
    #     Dense(1024=>512, bias = true), BatchNorm(512), x->elu(x,2),
    #     Dense(512=>512, x-> elu(x,2)),
    #     Dropout(0.3, active = true),
    #     Dense(512=> 2, x->elu(x,2))))
    	


    #printtree(mod)
    #return model
end

# Training function
function train(training_urls::Vector{<:ProductNode}, eval_urls::Vector{<:ProductNode}, valid_urls::Vector{<:ProductNode}, training_labels::Vector{Int64}, valid_labels::Vector{Int64}, 
    eval_labels::Vector{Int64}, param::Training_params{Int64}, prime::Int64)::Tuple{Vector{Float32}, ProductModel, Bool} 

    # Create model of the network
    model = create_model(param, prime) # What with the activation functions
    
    # Create a loss function
    loss(x, y) = Flux.Losses.logitbinarycrossentropy(x, OneHotArrays.onehotbatch(y .+ 1, 1:2))  # Onehot inside from training labels

    # Create dataloaders for training and validation
    training_data_loader = Flux.DataLoader((training_urls, training_labels), batchsize = param.batchsize, shuffle = true, partial = false)
    valid_data_loader = Flux.DataLoader((valid_urls, valid_labels), batchsize = param.batchsize, shuffle = true, partial = false)


    # Early stopping
    # Stopping  criteria for training loop
    state = Dict("best_loss" => Inf, "no_improv" => 0, "patience"=> 2)

    opt = Flux.Optimiser(Adam(param.learning_rate))
    opt_state = Flux.setup(opt, model)

    println("Learning rate is: ", param.learning_rate)
    # Early stopping callback
    function early_stop_cb()::Bool
        eval_loss = mean(loss(model(batch[1]), batch[2]) for batch in valid_data_loader)
        println("Validation loss is: ", eval_loss)
        if eval_loss < state["best_loss"]
            state["best_loss"] = eval_loss
            state["no_improv"] = 0
        else
            state["no_improv"] += 1
        end
        return state["no_improv"] > state["patience"]
    end

    stopped_at = false  # Early stop signalizer

    #15,true,256,2048,512,2048,512,2048,512,1024,0.001,0.781,0.742,0.76,0.697,6446,1811,2236,29507
    trainmode!(model)

    # Epoch loss
    epoch_losses = Float32[]
    numb_epochs = 0

    colors = ["red", "blue"]
    color_mapping = [colors[label] for label in training_labels.+1]
    # Training loop
    for i in 1:param.epochs
        @info "Epoch number $i."
        is_finite = true # Used for checking if loss is finite

        plot_data = softmax(model(training_urls))
        p_classes = scatter(plot_data[1,:], plot_data[2,:], color=color_mapping)
        display(p_classes)

        # Loss over time
        epoch_loss = 0
        batch_count = 0

        # Batch work
        for (x_batch, y_batch) in training_data_loader

            # Calculate the loss and gradients
            val, grads = Flux.withgradient(model) do m
                result = m(x_batch)
                batch_loss = loss(result, y_batch)
                epoch_loss +=batch_loss
                batch_count += 1
                return batch_loss
            end

            # Check for valid loss value
            if !isfinite(val)
                @warn "Loss is $val." 
                is_finite = false # Remark the loss
                break
            end
           
            # Update the parameters of the model (grads[1] is a nested set of NamedTuples)
            Flux.update!(opt_state, model, grads[1])
            
        end
        # Epochs
        numb_epochs += 1
        avg_epoch_loss = epoch_loss / batch_count
        push!(epoch_losses, avg_epoch_loss)
        

        # Early stopping
        if early_stop_cb()
            stopped_at = true
            break
        end
        # Control finite loss
        if !is_finite
            break
        end
        
    end
    testmode!(model) # Turn off dropout

    # Plot loss over epochs
    println(epoch_losses)
    println(numb_epochs)
    pl = plot(epoch_losses, 1:numb_epochs)
    display(pl)

    # Testing set performance
    println("Permormace on the training set:")
    train_probs = softmax(model(training_urls))[1,1:end]
    training_predicted = cold_treshold(train_probs, param.treshold)
    training_metric = evaluate_performance(training_labels, training_predicted)
    println("----------------")

    #println(softmax(model(eval_urls[1:50])))
    probs = softmax(model(eval_urls))[1,1:end]   # Transform output into probabilities

    # Return probabilities and trained model and if early early stopping occured
    return probs, model, stopped_at
end

# Cold treshold for converting a vector of probabilites into predictions
function cold_treshold(probs::Vector{Float32}, treshold::Float64 = 0.5)::Vector{Int64}
    return map(x-> x <= treshold ? 1 : 0, probs)
end

# Calculte the roc curve and pr curve, both are saved
function roc(probs::Vector{Float32}, true_labels::Vector{Int64}, filename::String)::Nothing
    treshold_range = 0.0:0.01:1.0    # Treshold range of values
    predicted_labels = map(x-> cold_treshold(probs, x), treshold_range)     # Vector of vectors of predicted labels for each treshold    
    tpr, fpr, prec = tpr_fpr(predicted_labels, true_labels)

    # Plot the ROC curve
    roc_curve = plot(fpr, tpr, label="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", legend=:bottomright)
    plot!(roc_curve,[0, 1], [0, 1], label="Random Classifier", linestyle=:dash)
    savefig(roc_curve, "ROC/" * filename * "_roc.pdf")

    # Plot the PR curve
    pr_curve = plot(tpr, prec, label="Precision/Recall Curve", xlabel="Recall", ylabel="Precision", legend=:bottomright, xlims=(0,1), ylims=(0,1))
    hline!(pr_curve, [0.5], label="Baseline Classifier", linestyle=:dash)
    savefig(pr_curve, "PR/" * filename * "_pr.pdf")
    return nothing
end

# Calculate the tpr, fpr and precision
function tpr_fpr(x::Vector{Vector{Int64}}, true_labels::Vector{Int64})::Tuple
    matricies = map(m->confusion_matrix(true_labels, m), x)
    tpr = zeros(Float64, length(x))
    fpr = zeros(Float64, length(x))
    prec = zeros(Float64, length(x))
    for i in 1:length(x)
        tpr[i] = matricies[i][1,1]/(matricies[i][1,1] + matricies[i][2,1]) # recall
        fpr[i] = matricies[i][1,2]/(matricies[i][1,2] + matricies[i][2,2])
        prec[i] = matricies[i][1,1]/(matricies[i][1,1] + matricies[i][1,2]) # precision
    end
    return tpr, fpr, prec
end

# Performs grid search and writes eval data into a csv file
function grid_search(training_urls::Vector{<:ProductNode}, eval_urls::Vector{<:ProductNode}, valid_urls::Vector{<:ProductNode}, training_labels::Vector{Int64}, eval_labels::Vector{Int64}, valid_labels::Vector{Int64}, training_param::Vector{Training_params{Int64, Float64}}, additional_param::Additional_params)::Nothing

    # Cycle for searching the grid
    @inbounds for i in 1:length(training_param)
        probs, model, stopped = train(training_urls, eval_urls, valid_urls, training_labels, valid_labels, eval_labels, training_param[i], additional_param.prime)   # train the network
        predicted_labels = cold_treshold(probs, training_param[i].treshold)     # Convert probabilities to labels based on treshold
        println("Performance on the evaluation set:")
        evaluation_metric = evaluate_performance(eval_labels, predicted_labels)    # Evaluate  performance with current network
        println("----------------")

        # Check if the model is "better"
        if evaluation_metric.Mathew > 0.7
            filename = "$(training_param[i].epochs)_$(training_param[i].batchsize)_$(training_param[i].learning_rate)_$(training_param[i].domain_neur_1)_$(training_param[i].domain_neur_2)_$(training_param[i].path_neur_1)_$(training_param[i].path_neur_2)_$(training_param[i].query_neur_1)_$(training_param[i].query_neur_2)_$(training_param[i].path_query_neur)"   # Construct filename
            roc(probs, eval_labels, filename)   # Calculate the ROC and PR curves and save them into a pdf
            #save("files/$filename.jld2", "Evaluation_metrics", evaluation_metric, "Training_params", training_param[i],"Additional_params", additional_param, "Model", model)   # Save the current network and its parameters 
        end
    
        # Save the model with state from jld2 - most robust
        s = Flux.state(model)
        jldsave("model.jld2", model_state=s) 
        
        # Create a row in the data frame for the current iteration
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
                       
        # Append the current row to the csv file
        CSV.write("Resresults.csv", data, append= true)
    end
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

# epochs, batchsize, treshold, domain_neur_1, domain_neur_2, path_neur_1, path_neur_2, query_neur_1, query_neur_2, path_query_neur, learning_rate
#ranges = [[15],  ct(POT(256,256)), ct(POT(512,512)), ct(POT(256,256)), ct(POT(1024,1024)),  ct(POT(512,512)), ct(POT(1024,1024)), ct(POT(512,512)), ct(POT(1024,1024)), [0.001], [0.5]]

ranges = [[15],  ct(POT(256,256)), ct(POT(2048,2048)), ct(POT(1024,1024)), ct(POT(2048,2048)),  ct(POT(1024,1024)), ct(POT(2048,2048)), ct(POT(1024,1024)), ct(POT(1024,1024)), [0.0005], [0.5]]

#ranges = [[10],  ct(POT(8,128)), ct(POT(1024,2048)), ct(POT(512,1024)), ct(POT(1024,2048)),  ct(POT(512,1024)), ct(POT(1024,2048)), ct(POT(512,1024)), ct(POT(1024,2048)), [0.001, 0.0005,0.0001]]

#println(size(filter_grid(par_grid(ranges))))

#15,true,128,1024,512,1024,512,1024,512,1024,0.0005,0.851,0.705,0.77,0.72,6094,1069,2546,30291

main(ranges, 300)

