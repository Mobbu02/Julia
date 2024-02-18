using CSV, DataFrames, Flux, Mill, URIs, Cthulhu, ProfileView, BenchmarkTools, JLD2, Shuffle, OneHotArrays

parts_of_url = ["query", "path"]
function main(prime::Int64, neurons::Int64)
    # Read lines
    all_lines = readlines("mixed_lines.txt") 
    #shuffle(collect(1:round(Int64, 1/5*length(all_lines))))
    new_order_of_lines = collect(1:round(Int64, 1/5*length(all_lines)))         # Collect all indicies and shuffle them

    lines = all_lines[new_order_of_lines]           # Choose new lines

    labels = load("labels.jld2", "labels")      # Load labels for instances

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
    
    #fill_dict(dict, bag_indx_parts, lines)      # Fill dictionary 
    ProfileView.@profview fill_dict(dict, bag_indx_parts, lines)
    #descend(fill_dict,Tuple{Dict{String, Vector{Vector{String}}}, Dict{String, Vector{Int64}}, Vector{String}})
    #@descend fill_dict(dict, bag_indx_parts, lines)
    
    for key in parts_of_url         # For every key
        for i=1:length(dict[key])       # For every vector in that value vector
            dict_hashed[key][i] = transform_to_trig(dict[key][i], prime)       # Transform vector in a 
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
            histogram_dict[key] = hashed_to_hist.(dict_hashed[key], prime)             # Transform hashed trigrams into histograms 
    end   

    # Creating bags from from histograms
    bag_path = create_bags(histogram_dict["path"], "path", bag_indx_parts, prime)
    bag_query = create_bags(histogram_dict["query"], "query", bag_indx_parts, prime)

    bag_path_index = create_range(bag_indx_parts["path"])
    bag_query_index = create_range(bag_indx_parts["query"])

    # Train
    train(bag_path, bag_query, bag_path_index, bag_query_index, labels, neurons)
end


# Fill dict with URL parts and fill bag_indx_parts with lenghts of parts of url
function fill_dict(D::Dict{String, Vector{Vector{String}}}, E::Dict{String, Vector{Int64}}, lines::Vector{String})
    D[parts_of_url[2]] = URIs.splitpath.(URI.(lines))           # Map vectors of strings into a D["path"] vector
    D[parts_of_url[1]] = vectorize_queryparams.(URIs.queryparams.(URI.(lines)))             # Map vectors of strings into a D["query"] vector

    # Alocate memory for E dict = bag_indx_parts
    E[parts_of_url[1]] = length.(D[parts_of_url[1]])
    E[parts_of_url[2]] = length.(D[parts_of_url[2]])
end

# Creating a string vector of key+value from dictionary
function vectorize_queryparams(X::Dict{String, String})::Vector{String}
    return map(key -> key * "=" * X[key],collect(keys(X)))
end


# Transforming vector of words into a vector of trigrams
function transform_to_trig(x::Vector{String}, prime::Int64)::Vector{Vector{UInt64}}
    trig_vec = Vector{Vector{UInt64}}(undef, length(x))         # Initialize vector to a length of the whole string
    # Initialize internal vectors
    for i = 1:length(trig_vec)
        trig_vec[i] = Vector{UInt64}(undef, length(x[i])-2)         # For each vector within - Initialize new vector to a length of string
    end

    # For every string in x, create trigram, hash it, map it into a vector on position[ind]
    for (ind, val) in enumerate(x)
        map!(trig->trig%prime, trig_vec[ind], hash.([val[i:i+2] for i=1:length(val)-2]))         # Create trigrams, hash them, modulo each hash
    end
    return trig_vec
end

# Place number into a vector - index corresponding to its value
function place_numb(x::Vector{Int64}, numb::UInt64)
    x[Int64(numb)] += 1
end

# Mapping hashed vector onto a histogram vector
function hashed_to_hist(x::Vector{Vector{UInt64}}, prime::Int64)::Vector{Vector{Int64}}
    vec_to_ret = Vector{Vector{Int64}}(undef, length(x))        # Initialize the vector to return
    map!( y->begin # Vector{UInt64}         # Creating trigrams and mapping them into another vector
        mapping_vec = zeros(Int64, prime)  
        map(val-> val != 0 && place_numb(mapping_vec, val), y)       # Place numbers onto their corresponding place in the new vector
        return mapping_vec
    end, vec_to_ret, x)
    return vec_to_ret
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


function train(bag_path::Matrix{UInt64}, bag_query::Matrix{UInt64}, bag_path_index::Vector{UnitRange{Int64}}, bag_query_index::Vector{UnitRange{Int64}}, y::Vector{Int64}, neurons::Int64)
     # Structure of Node
     ds = BagNode(ProductNode((BagNode(bag_path,
                                        bag_path_index),
                                BagNode(bag_query, 
                                bag_query_index))), # Won't work, because I deleted 0s 
                    [1:1, 2:2])

    printtree(ds)

    
    y = map(i->maximum(y[i])+1, ds.bags)
    y_oh = onehotbatch(y,1:2)

    # Model through reflection model 
    model = reflectinmodel(ds, d->Dense(d,neurons), SegmentedMeanMax)
    printtree(model)

    model(ds)

    opt_state = Flux.setup(Adam(), model)


    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    # for e in 1:500
    #     if e % 10 == 1
    #         @info "Epoch $e" training_loss=loss(model, ds, y_oh)
    #     end
    #     Flux.train!(loss, model, [(ds, y_oh)], opt_state)
    # end
end
   
main(2053,2)