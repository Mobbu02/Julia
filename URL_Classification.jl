using Revise
using Profile, DataFrames, Flux, Mill, URIs, ProfileView, BenchmarkTools, JLD2, OneHotArrays, Random, Serialization
#, CSV, Cthulhu,

include("dictionary_handling.jl")
include("eval_mod.jl")
using .dictionary_handling
using .eval_mod

parts_of_url = ["query", "path"]
function main(prime::Int64, ND1_path::Int64, ND1_query::Int64, ND2_path::Int64, ND2_query::Int64, NP_1::Int64)
    # Read lines
    all_lines = load("JLD2_dicts&lines/mixed_lines.jld2", "mixed_lines")       #  Read URL addresses
    all_labels = load("JLD2_dicts&lines/labels.jld2", "labels")      # Load labels for instances
    

    random_seed = rand(UInt32)      # Generate random number
    println(random_seed)
    new_indexes = randcycle(MersenneTwister(random_seed), length(all_lines))        # Generate permutation of set length, seed the generator with random_seed

    lines = all_lines[new_indexes[1:Int(round(1/50*length(new_indexes)))]]          # Choose new lines
    labels = all_labels[new_indexes[1:Int(round(1/50*length(new_indexes)))]]        # Choose new labels

    size_to_alloc = size(lines, 1)      # Amount of lines loaded

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
    dictionary_handling.fill_dict!(dict, bag_indx_parts, size_to_alloc, lines)      # Fill dict & bag_indx_parts
    dictionary_handling.fill_dict_hashed!(dict, dict_hashed, size_to_alloc, prime)     # Fill Dict_hashed
    dictionary_handling.fill_hash_dict!(dict_hashed, histogram_dict, size_to_alloc, prime)     # Fill histogram_dict

    # Creating bags from histograms
    bag_path = create_bags(histogram_dict["path"], "path", bag_indx_parts, prime)
    bag_query = create_bags(histogram_dict["query"], "query", bag_indx_parts, prime)

    bag_path_index = create_range(bag_indx_parts["path"])
    bag_query_index = create_range(bag_indx_parts["query"])

    neurons = tuple(ND1_path, ND1_query, ND2_path, ND2_query, NP_1)

    # Train
    predicted = train(bag_path, bag_query, bag_path_index, bag_query_index, labels, prime, 2, 2, 4, 4, 4)
    eval_mod.eval_model("Model_eval.txt", predicted, labels, Int64(random_seed), neurons)
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

# Create a non-overlapping vector of ranges
function create_range(x::Vector{Int64})::Vector{UnitRange{Int64}}
    """
    Empty vector  = url does not have a query is recorded as an empty range, for example: 13:15, 15:15, 16:17. 
    15:15 indicates an empty vector
    """
    start = 1
    ranges = map(y-> begin
        rg = start:(start+y-1)
        if y==0
            rg = (start-1):(start-1)
        end
        start += y
        return rg
    end, x)
    return ranges
end

# Could be used in train functin?
# struct Neuron_params
#     ND1_path::Int64
#     ND1_query::Int64
#     ND2_path::Int64
#     ND2_query::Int64
#     NP_1::Int64
# end

function train(bag_path::Matrix{UInt64}, bag_query::Matrix{UInt64}, bag_path_index::Vector{UnitRange{Int64}}, bag_query_index::Vector{UnitRange{Int64}}, 
    y::Vector{Int64}, prime::Int64, ND1_path::Int64, ND1_query::Int64, ND2_path::Int64, ND2_query::Int64, NP_1::Int64)

     # Structure of Node
     ds = ProductNode((BagNode(bag_path,
                                        bag_path_index),
                                BagNode(bag_query, 
                                bag_query_index)))

    # OneHot
    y = map(i->maximum(y[i])+1, y.+1)
    y_oh = onehotbatch(y,1:2)

    model = ProductModel(tuple(
        BagModel(
            ArrayModel(Dense(prime=>ND1_path)), SegmentedMeanMax(ND1_path), Dense(2*ND1_path=>ND2_path)),
        BagModel(
            ArrayModel(Dense(prime=>ND1_query)), SegmentedMeanMax(ND1_query), Dense(2*ND1_query=>ND2_query))
        ),  Chain(Dense(ND2_path+ND2_query=>NP_1), Dense(NP_1=>2), softmax))

    # Optimum
    opt_state = Flux.setup(Adam(), model)

    # Loss function
    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    # Training loop
    for i=1:100
        # if i % 10 ==1
        #     @info "Epoch $i" training_loss=loss(model, ds, y_oh)
        # end
        Flux.train!(loss, model, [(ds, y_oh)], opt_state)
    end

    predicted = model(ds) # Should act on new data 
    return predicted
end
   
main(2053, 6, 2, 5, 1, 4)

