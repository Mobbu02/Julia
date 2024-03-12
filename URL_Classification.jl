using Revise
using  Flux, Mill, JLD2, OneHotArrays, Random, Statistics
#, CSV, Cthulhu,

include("dictionary_handling.jl")
include("eval_mod.jl")
using .dictionary_handling
using .eval_mod


function main(prime::Int64, ND1_path::Int64, ND1_query::Int64, ND2_path::Int64, ND2_query::Int64, NP_1::Int64, epoch::Int64, batchsize::Int64)
    # Read lines
    all_lines = load("Julia/Lines/new_combined_lines.jld2", "lines")      #  Read URL addresses
    all_labels = load("Julia/Lines/shuffled_labels.jld2", "labels_vector")      # Load labels for instances
    
    random_seed = rand(UInt32)      # Generate random number
    println(random_seed)
    new_indexes = randcycle(MersenneTwister(random_seed), length(all_lines))        # Generate permutation of set length, seed the generator with random_seed
    number_of_training_indexes = Int(round(1/100*length(new_indexes)))
    
    training_lines = all_lines[new_indexes[1:number_of_training_indexes]]          # Choose training lines
    training_labels = all_labels[new_indexes[1:number_of_training_indexes]]        # Choose training labels

    eval_lines = all_lines[new_indexes[number_of_training_indexes:2*number_of_training_indexes]]        # Choose eval lines
    eval_labels = all_labels[new_indexes[number_of_training_indexes:2*number_of_training_indexes]]      # Choose eval labels

    per_of_positive_instances = round((sum(eval_labels))/length(eval_labels), digits = 2)       # Calculate the amount of positive URLs included within the training set
    println("Percentage of positive urls within the loaded lines: ", per_of_positive_instances)
    
    # Prepare training data
    path_train, query_train, path_index_train, query_index_train = prepare_data(prime, training_lines)

    # Prepare eval data
    path_eval, query_eval, path_index_eval, query_index_eval = prepare_data(prime, eval_lines)

    # Neurons + additional info
    network_info = tuple(ND1_path, ND1_query, ND2_path, ND2_query, NP_1, epoch, per_of_positive_instances, batchsize)

    # Train
    predicted = train(path_train, query_train, path_index_train, query_index_train, training_labels, prime, ND1_path, ND1_query, ND1_path, ND2_query, NP_1, path_eval, query_eval, path_index_eval, query_index_eval, epoch, batchsize, eval_labels)
    println(predicted)
    #eval_mod.eval_model("Model_eval.txt", predicted, eval_labels, Int64(random_seed), network_info)
end
    
function prepare_data(prime::Int64, lines::Vector{String})::Tuple{Matrix{UInt64}, Matrix{UInt64}, Vector{UnitRange{Int64}}, Vector{UnitRange{Int64}}}
    # Parts of the URL
    parts_of_url = ["query", "path"]

    # Dictionary for storage of part of URLS (Remainder: dictionaries automatically resize) 
    dict = Dict{String,Vector{Vector{String}}}()
    dict[parts_of_url[1]] = []      
    dict[parts_of_url[2]] = []
    
    # Dictionary for storing hashed and moduled trigrams
    dict_hashed = Dict{String,Vector{Vector{Vector{UInt64}}}}()
    dict_hashed[parts_of_url[1]] = []
    dict_hashed[parts_of_url[2]] = []
    
    # Dictionary for storing the lenghts of parts of URLs
    bag_indx_parts = Dict{String, Vector{Int64}}()
    bag_indx_parts[parts_of_url[1]] = []
    bag_indx_parts[parts_of_url[2]] = []

    # Dictionary of histogram vectors
    histogram_dict = Dict{String, Vector{Vector{Vector{Int64}}}}()
    histogram_dict[parts_of_url[1]] = []
    histogram_dict[parts_of_url[2]] = []

    # Fill dictionaries
    dictionary_handling.fill_dict!(dict, bag_indx_parts, lines)      # Fill dict & bag_indx_parts
    dictionary_handling.fill_dict_hashed!(dict, dict_hashed, prime)     # Fill Dict_hashed
    dictionary_handling.fill_hash_dict!(dict_hashed, histogram_dict, prime)     # Fill histogram_dict

    # Creating bags from histograms
    bag_path = create_bags(histogram_dict["path"], "path", bag_indx_parts, prime)
    bag_query = create_bags(histogram_dict["query"], "query", bag_indx_parts, prime)    
    bag_path_index = create_range(bag_indx_parts["path"])
    bag_query_index = create_range(bag_indx_parts["query"])

    return bag_path, bag_query, bag_path_index, bag_query_index
end 


function count_size(key::String, B::Dict{String, Vector{Int64}})::Int64
    return sum(B[key]) + count(x->x==0, B[key])
end

# Create bags(Array Node) - matrix of vectors
function create_bags(D::Vector{Vector{Vector{Int64}}}, key::String, B::Dict{String, Vector{Int64}}, prime::Int64)::Matrix{UInt64}   
    matrix_to_return = zeros(UInt64, prime, count_size(key, B))

    # Assigning vectors to columns of matrix_to_return
    for i in eachindex(D)
        column_index = 1
        for vec in D[i]
            matrix_to_return[:,column_index] .= vec
            column_index +=1
        end
    end
    return matrix_to_return
end

function create_range(x::Vector{Int64})::Vector{UnitRange{Int64}}
    """
    Create a vector of non-overlapping integer ranges based on the input vector. Each element in the input vector 'x' 
    represents the length of the range. A '0' in 'x' results in an empty range (1:0).
    """
    start = 1
    ranges = map(y -> begin
        rg = start:(start + max(y - 1, 0))
        start += max(y, 1)  # Ensure `start` is incremented at least by 1 to avoid overlapping ranges.
        return rg
    end, x)
    return ranges
end

function create_node_model(bag_path::Matrix{UInt64}, bag_query::Matrix{UInt64}, bag_path_index::Vector{UnitRange{Int64}}, 
    bag_query_index::Vector{UnitRange{Int64}})
    return (ProductNode(tuple(BagNode(bag_path,
                                bag_path_index),
                        BagNode(bag_query, 
                            bag_query_index)))
                            )
end

# Train function 
function train(bag_path_training::Matrix{UInt64}, bag_query_training::Matrix{UInt64}, bag_path_index_training::Vector{UnitRange{Int64}}, bag_query_index_training::Vector{UnitRange{Int64}}, 
    y_train::Vector{Int64}, prime::Int64, ND1_path::Int64, ND1_query::Int64, ND2_path::Int64, ND2_query::Int64, NP_1::Int64,
    bag_path_eval::Matrix{UInt64}, bag_query_eval::Matrix{UInt64}, bag_path_index_eval::Vector{UnitRange{Int64}}, bag_query_index_eval::Vector{UnitRange{Int64}}, epoch::Int64, batchsize::Int64, eval_labels)

    # Create node model 
    ds = create_node_model(bag_path_training, bag_query_training, bag_path_index_training, bag_query_index_training)

    # OneHot
    #y_train = map(i->maximum(y_train[i])+1, y_train.+1) # Tohle dela blbost
    #y_oh = OneHotArrays.onehotbatch(y_train,1:2)

    # Structure of the model
    model = ProductModel(tuple(
        BagModel(
            ArrayModel(Dense(prime=>ND1_path, bias = true, tanh)), SegmentedMeanMax(ND1_path), Dense(2*ND1_path=>ND2_path,bias =true, relu)),
        BagModel(
            ArrayModel(Dense(prime=>ND1_query, bias = true, tanh)), SegmentedMeanMax(ND1_query), Dense(2*ND1_query=>ND2_query, bias = true, relu))
        ),  Chain(Dense(ND2_path+ND2_query=>NP_1, bias = true, tanh), Dense(NP_1=>2), softmax))


    # Optimum
    opt_state = Flux.setup(Adam(), model)

    # Loss function
    loss(m, x, y) = Flux.logitcrossentropy(m(ProductNode(x)), OneHotArrays.onehotbatch(y .+ 1, 1:2))
    acc(ds, y) = Statistics.mean(onecold(model(ds)) .== y.+1)

    # Prepare data, shouldn't be onehot encode
    train_loader = Flux.DataLoader((data=ds.data, labels=y_train), batchsize = batchsize, shuffle = true, partial = false) # Batchsize shouldn't be equal to the number of url, is the data input correct?

    model_to_eval = create_node_model(bag_path_eval, bag_query_eval, bag_path_index_eval, bag_query_index_eval)         # Create Node model for new data 

    # Training loop
    for i in 1:epoch
        for (x,z) in train_loader
            Flux.train!(loss, model, [(x,z)], opt_state) # Not sure if it's learning from the correct stuff
        end
        if i % 10 == 1
            @show acc(model_to_eval, eval_labels)
        end
    end


    predicted = model(model_to_eval)        # Predict labels for non-training data
    return predicted
end
 
main(2053, 20, 60, 10, 60, 30, 60, 32)
#ProfileView.@profview main(2053, 30, 30, 10, 10, 8)

