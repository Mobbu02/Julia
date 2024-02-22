using CSV, DataFrames, Flux, Mill, URIs, Cthulhu, ProfileView, BenchmarkTools, JLD2, OneHotArrays, Random


include("dictionary_handling.jl")
using .dictionary_handling

parts_of_url = ["query", "path"]
function main(prime::Int64, neurons::Int64)
    # Read lines
    all_lines = load("mixed_lines.jld2", "mixed_lines")       #  Read URL addresses
    all_labels = load("labels.jld2", "labels")      # Load labels for instances
    
    new_indexes = randcycle(MersenneTwister(), length(all_lines))        # Generate permutation of set length
    

    lines = all_lines[new_indexes[1:Int(round(1/5*length(new_indexes)))]]          # Choose new lines
    labels = all_labels[new_indexes[1:Int(round(1/5*length(new_indexes)))]]        # Choose new labels
    
    size_to_alloc = size(lines, 1)      # Amount of lines loaded

    # Dictionary for storage of part of URLS (Remainder: dictionaries automatically resize) 
    dict = Dict{String,Vector{Vector{String}}}()
    dict[parts_of_url[1]] = []
    dict[parts_of_url[2]] = []
    
    # Fill dict values with vector of vectors of strings
    for key in parts_of_url
        dict[key] = Vector{Vector{String}}(undef, size_to_alloc) 
    end
    
    # Dictionary for storing hashed and moduled trigrams
    # Allocated in fill_dict after loading original parts of URL
    dict_hashed = Dict{String,Vector{Vector{Vector{UInt64}}}}()
    dict_hashed[parts_of_url[1]] = []
    dict_hashed[parts_of_url[2]] = []
    
    # Fill dict_hashed with vector of vector of vectors of strings
    for key in parts_of_url
        dict_hashed[key] = Vector{Vector{String}}(undef, size_to_alloc) 
    end
    
    # Dictionary for storing the lenghts of parts of URLs
    bag_indx_parts = Dict{String, Vector{Int64}}()
    bag_indx_parts[parts_of_url[1]] = []
    bag_indx_parts[parts_of_url[2]] = []
    
    dictionary_handling.fill_dict(dict, bag_indx_parts, lines)      # Fill dictionary 
    
    for key in parts_of_url         # For every key
        for i=1:length(dict[key])       # For every vector in that value vector
            dict_hashed[key][i] = dictionary_handling.transform_to_trig(dict[key][i], prime)       # Transform vectors in dict
        end
    end
    
    # Dictionary of histogram vectors
    histogram_dict = Dict{String, Vector{Vector{Vector{Int64}}}}()          # This will store vectors, where each one represents a histogram
    histogram_dict[parts_of_url[1]] = []
    histogram_dict[parts_of_url[2]] = []
    
    # Preallocate
    for key in parts_of_url
        histogram_dict[key] = Vector{Vector{Int64}}(undef, size_to_alloc) 
    end
     
    for key in parts_of_url
            histogram_dict[key] = dictionary_handling.hashed_to_hist.(dict_hashed[key], prime)             # Transform hashed trigrams into histograms 
    end   

    # Creating bags from from histograms
    bag_path = create_bags(histogram_dict["path"], "path", bag_indx_parts, prime)
    bag_query = create_bags(histogram_dict["query"], "query", bag_indx_parts, prime)

    bag_path_index = create_range(bag_indx_parts["path"])
    bag_query_index = create_range(bag_indx_parts["query"])
    #println(size(bag_path_index))
    #println(size(bag_query_index))
    #println(size(bag_query))
    #println(size(bag_path))

    # Train
    train(bag_path, bag_query, bag_path_index, bag_query_index, labels, neurons, prime)
end

function count_size(key::String, B::Dict{String, Vector{Int64}})::Int64
    return sum(B[key]) + count(x->x==0, B[key])
end

# Create bags(Array Node) - matrix of vectors
function create_bags(D::Vector{Vector{Vector{Int64}}}, key::String, B::Dict{String, Vector{Int64}}, prime::Int64)::Matrix{UInt64}
    #matrix_to_return = Matrix{UInt64}(undef, 2053, count_size(key, B))         # Pzn. U prazdneho vektoru je tam garbage - sum pres vektor da velke cislo misto nuly
    matrix_to_return = zeros(UInt64, prime, count_size(key, B))

    # Assigning vectors to columns of matrix_to_return
    for i=1:length(D)
        column_indx = 1
        for vec in D[i]
            matrix_to_return[:,column_indx] .= vec
            column_indx += 1
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


function train(bag_path::Matrix{UInt64}, bag_query::Matrix{UInt64}, bag_path_index::Vector{UnitRange{Int64}}, bag_query_index::Vector{UnitRange{Int64}}, y::Vector{Int64}, neurons::Int64, prime::Int64)
     # Structure of Node
    #  ds = BagNode(ProductNode((BagNode(bag_path,
    #                                     bag_path_index),
    #                             BagNode(bag_query, 
    #                             bag_query_index))),
    #                 [1:1, 2:2]) #   Tohle musí být špatně. Musí zde být označení instancí, které jsou positiv a negativ?

    BM1 = BagNode(bag_path, bag_path_index)
    BM2 = BagNode(bag_query, bag_query_index)
    PN = tuple(BM1, BM2) |> ProductNode
    ds = BagNode(PN, [1:1, 2:2])
    printtree(ds)


    #println(ds.bags) # = [1:1, 2:2]

    # OneHot
    y = map(i->maximum(y[i])+1, ds.bags) # Tohle je dobře, ale pracuje to se špatnými daty!
    #println(size(y))
    y_oh = onehotbatch(y,1:2)
    #println(y_oh)


    # Model    
    model = BagModel(           # totaly same model as from reflectinmodel
                ProductModel(tuple(
                    BagModel(
                        ArrayModel(Dense(prime=>2)), SegmentedMeanMax(2), Dense(4=>4)),
                    BagModel(
                        ArrayModel(Dense(prime=>2)), SegmentedMeanMax(2), Dense(4=>4))
                    ),  Dense(8=>4)),
            SegmentedMeanMax(4), Chain(Dense(8=>2), softmax))  # Output has to be 2 bcs I am classyfing into 2 classes right?
    
    #printtree(model)
    
    print(model(ds))
    opt_state = Flux.setup(Adam(), model)


    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    # for e in 1:10
    #     if e % 10 == 1
    #         @info "Epoch $e" training_loss=loss(model, ds, y_oh)
    #         #acc = mean(Flux.onecold(model(ds), 1:2) .==y)
    #         #print(acc)
    #     end
    #     Flux.train!(loss, model, [(ds, y_oh)], opt_state)
    # end
end
   
main(2053,2)