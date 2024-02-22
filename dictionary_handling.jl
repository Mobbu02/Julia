# Module for handling the creation of needed dictionaries and operations with them
module dictionary_handling

using URIs

export fill_dict
export vectorize_queryparams
export transform_to_trig
export place_numb
export hashed_to_hist

parts_of_url = ["query", "path"]

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


end