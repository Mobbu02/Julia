using CSV, DataFrames, Flux, Mill, URIs, Cthulhu, ProfileView, BenchmarkTools, JLD2

parts_of_url = ["query", "path"]
function main()
    # Read lines
    lines = readlines("mixed_lines.txt") # TODO: Read only 20% of lines 
    size_to_alloc = size(lines)[1] # Amount of lines loaded
    
    # Dictionary for storage of part of URLS (Remainder: dictionaries automatically resize) 
    dict = Dict{String,Vector{Vector{String}}}()
    dict["query"] = []
    dict["path"] = []
    
    # Fill dict values with vector of vectors of strings
    for key in parts_of_url
        dict[key] = Vector{Vector{String}}(undef, size_to_alloc) 
    end
    
    # Dictionary for storing hashed and moduled trigrams
    # Allocated in fill_dict after loading original parts of URL
    dict_hashed = Dict{String,Vector{Vector{Vector{UInt64}}}}()
    dict_hashed["query"] = []
    dict_hashed["path"] = []
    
    # Fill dict_hashed with vector of vector of vectors of strings
    for key in parts_of_url
        dict_hashed[key] = Vector{Vector{String}}(undef, size_to_alloc) 
    end
    
    # Dictionary for storing the lenghts of parts of URLs
    bag_indx_parts = Dict{String, Vector{Int64}}()
    bag_indx_parts["query"] = []
    bag_indx_parts["path"] = []
    
    fill_dict(dict, bag_indx_parts, lines)      # Fill dictionary 
    #ProfileView.@profview fill_dict(dict, bag_indx_parts, lines)
    #descend(fill_dict,Tuple{Dict{String, Vector{Vector{String}}}, Dict{String, Vector{Int64}}, Vector{String}})
    #@descend fill_dict(dict, bag_indx_parts, lines)
    
    for key in parts_of_url         # For every key
        for i=1:length(dict[key])       # For every vector in that value vector
            dict_hashed[key][i] = transform_to_trig(dict[key][i])       # Transform vector in a 
        end
    end
    
    # Dictionary of histogram vectors
    histogram_dict = Dict{String, Vector{Vector{Vector{Int64}}}}()
    histogram_dict["query"] = []
    histogram_dict["path"] = []
    
    # Preallocate
    for key in parts_of_url
        histogram_dict[key] = Vector{Vector{Int64}}(undef, size_to_alloc) 
    end
    
    for key in parts_of_url
        for i=1:length(histogram_dict[key])
            histogram_dict[key][i] = hashed_to_hist(dict_hashed[key][i])
        end
    end   
    bag_path = create_bags(histogram_dict["path"], "path", bag_indx_parts)
    bag_query = create_bags(histogram_dict["query"], "query", bag_indx_parts)

    bag_path_index = create_range(bag_indx_parts["path"])
    bag_query_index = create_range(bag_indx_parts["query"])

    
    train(bag_path, bag_query, bag_path_index, bag_query_index)

   

end


# Fill dict with URL parts and fill bag_indx_parts with lenghts of parts of url
function fill_dict(D::Dict{String, Vector{Vector{String}}}, E::Dict{String, Vector{Int64}}, lines::Vector{String})

    map!(vecs->vecs, D["path"], URIs.splitpath.(URI.(lines)))       # Map vectors of strings into a D["path"] vector
    map!(vecs->vecs, D["query"], vectorize_queryparams.(URIs.queryparams.(URI.(lines))))        # Map vectors of strings into a D["query"] vector

    # Alocate memory for E dict = bag_indx_parts
    # Map lengths of parts of URLs into a dictionary vector
    E["path"] = Vector{Int64}(undef, length(D["path"]))
    map!(numb->numb, E["path"], length.(D["path"]))     # Map lenghts of D["path"] vectors to another vector

    E["query"] = Vector{Int64}(undef, length(D["query"]))
    map!(numb->numb, E["query"], length.(D["query"])) # Map lenghts of D["query"] vectors to another vector
end

# Creating a string vector of key+value from dictionary
function vectorize_queryparams(X::Dict{String, String})
    positions = Vector{String}(undef, length(X))
    """
    collect(keys(X)) returns a vector of keys from the dictionary x
    Annonymous function creates key + X[key] string and this inserted into a vector_positions
    """
    map!(key -> string(key, "=", X[key]), positions, collect(keys(X))) 
    return positions
end


# Transforming vector of words into a vector of trigrams
function transform_to_trig(x::Vector{String})
    trig_vec = Vector{Vector{UInt64}}(undef, length(x))         # Initialize vector to a length of the whole string
    # Initialize internal vectors 
    for i = 1:length(trig_vec)
        trig_vec[i] = Vector{UInt64}(undef, length(x[i])-2)         # for each vector within - Initialize to length of string
    end

    # For every string in x, create trigram, hash it, map it into a vector on position[ind]
    for (ind, val) in enumerate(x)
        map!(trig->trig%2053, trig_vec[ind], hash.([val[i:i+2] for i=1:length(val)-2]))         # Create trigrams, hash them, modulo each hash
    end
    return trig_vec
end

# Place number into a vector - index corresponding to its value
function place_numb(x::Vector{Int64}, numb::UInt64) 
    x[Int64(numb)] += 1
end

# Mapping hashed vector onto a histogram vector
function hashed_to_hist(x::Vector{Vector{UInt64}})
    vec_to_ret = Vector{Vector{Int64}}(undef, length(x)) # Initialize the vector to return
    
    map!( y->begin # Vector{UInt64}         # Creating trigrams and mapping them into another vector
        mapping_vec = zeros(Int64, 2053)  
        map(val-> val != 0 && place_numb(mapping_vec, val), y)       # Place numbers onto their corresponding placei in the new vector
        return mapping_vec
    end, vec_to_ret, x)
    return vec_to_ret
end

function count_size(key::String, B::Dict{String, Vector{Int64}})
    return sum(B[key])
end

# Create bags(Array Node) - matrix of vectors
function create_bags(D::Vector{Vector{Vector{Int64}}}, key::String, B::Dict{String, Vector{Int64}})
    matrix_to_return = Matrix{UInt64}(undef, 2053, count_size(key, B)) 
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
function create_range(x::Vector{Int64}) 
    numb_of_ranges = count(x->x>0, x) # Count the amount non zero numbers
    y = filter(numb-> numb!=0,x) # Only non zero values
    vector_to_ret = Vector{UnitRange{Int64}}(undef, numb_of_ranges) # Create a vector of numb_of_ranges size
    start = 1
    # Map ranges into a vector_to_ret
    map!(numb-> begin 
         to_ret = start:(start+numb-1) 
         start+=numb
         return to_ret
         end,vector_to_ret,y)
    return vector_to_ret
end


function train(bag_path::Matrix{UInt64}, bag_query::Matrix{UInt64}, bag_path_index::Vector{UnitRange{Int64}}, bag_query_index::Vector{UnitRange{Int64}})
     # Structure of Node
     ds = BagNode(ProductNode((BagNode(bag_path,
                                        bag_path_index),
                                BagNode(bag_query, 
                                bag_query_index))), # Won't work, because I deleted 0s 
                    [1:1, 2:2])

    #printtree(ds)

    # Model through reflection model 
    model = reflectinmodel(ds, d->Dense(d,8), SegmentedMeanMax)
    #printtree(model)

    model(ds)

    opt_state = Flux.setup(Adam(), model)


    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    for e in 1:100
        if e % 10 == 1
            @info "Epoch $e" training_loss=loss(model, ds, y_oh)
        end
        Flux.train!(loss, model, [(ds, y_oh)], opt_state)
    end
end
   
main()