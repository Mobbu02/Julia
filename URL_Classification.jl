using CSV, DataFrames, Flux, Mill, URIs, Cthulhu, ProfileView

parts_of_url = ["fragment", "query", "path"]
function main()

    # Read lines
    lines = readlines("mixed_lines.txt") # TODO: Read only 20% of lines 
    size_to_alloc = size(lines)[1] # Amount of lines loaded
    print(lines[1])
    ## TODO: Prelalocate sizes 
    # Dictionary for storage of part of URLS (Remainder: dictionaries automatically resize) 
    dict = Dict{String,Vector{Vector{String}}}()
    dict["fragment"] = []
    dict["query"] = []
    dict["path"] = []

    # Prealocating size for the Vector of Vectors of Strings
    for key in parts_of_url
        dict[key] = Vector{Vector{String}}(undef, size_to_alloc)
    end


    # TODO: Prealocate size for the rest
    dict_hashed = Dict{String,Vector{Vector{Vector{UInt64}}}}()
    dict_hashed["fragment"] = []
    dict_hashed["query"] = []
    dict_hashed["path"] = []


    bag_indx_parts = Dict{String, Vector{Int64}}()
    bag_indx_parts["fragment"] = []
    bag_indx_parts["query"] = []
    bag_indx_parts["path"] = []
end


main()



# TODO: Remove pushes
# Add url to dict and add lengths of part of the url to the bag_indx_parts
function add_url_to_dict(x::URI, dict::Dict{String, Vector{Vector{String}}}, indx::Dict{String, Vector{Int64}})
    path = string.(URIs.splitpath(x.path))
    push!(get!(dict, "path", Vector{String}), path)
    push!(indx["path"], length(path))
    
    query = [string(pair[1], "=", pair[2]) for pair in pairs(queryparams(x))]
    push!(get!(dict, "query", Vector{string}), query)
    push!(indx["query"], length(keys(queryparams(x))))
    
    fragment = string.(x.fragment)
    push!(get!(dict, "fragment", Vector{string}), [fragment])
    push!(indx["fragment"], length(fragment)) 
end


# Transforming vector of words into a vector of trigrams
function transform_to_trig(x::Vector{String})
    trig_vec = Vector{Vector{UInt64}}()
    for elem in x
        cur = [elem[i:i+2] for i=1:length(elem)-2]
        a = map(hash,cur).%2053 #hash trigrams and modulo them
        push!(trig_vec, a)
    end
    return trig_vec
end


# Mapping hashed vector onto a histogram vector
function map_to_vec(x::Vector{UInt64})
    map_vec = zeros(UInt64, 2053)
    for numb in x
        if numb==0
            continue
        end
        map_vec[numb] += 1
    end
    return map_vec
end

# Fill dict with URL parts and fill bag_indx_parts with lenghts of parts of url
function fill_dict(D::Dict{String, Vector{Vector{String}}}, E::Dict{String, Vector{Int}}, lines::Vector{String})
    



    for line in lines
        add_url_to_dict(URI(line), D, E)
        i += 1
        if(i > 10)
            break
        end
    end
end

#ProfileView.@profview fill_dict(dict, bag_indx_parts,"mixed_lines.txt")
#fill_dict(dict, bag_indx_parts,"mixed_lines.txt")
#print(dict)
function trig_map(D::Dict{String, Vector{Vector{String}}}, E::Dict{String, Vector{Vector{Vector{UInt64}}}})
    # Keep track of lengths of part of URL somehow
    for key in keys(D)
        for i=1:length(D[key])
            if(isempty(D[key][i]))
                    push!(E[key], [[0]])
                    continue 
            end
            push!(E[key], transform_to_trig(D[key][i])) # transfrom to trig every single vector
        end
    end
end
#trig_map(dict, dict_hashed)

# Transform to hist. vector
function to_hist_vec(D::Dict{String,Vector{Vector{Vector{UInt64}}}})
    for key in keys(D)
        for i=1:length(D[key])
            for j=1:length(D[key][i])
                 D[key][i][j] = map_to_vec(D[key][i][j])
            end
        end
    end
end


#to_hist_vec(dict_hashed)

function count_size(key::String)
    return sum(bag_indx_parts[key])
end

function create_bags(D::Vector{Vector{Vector{UInt64}}}, key::String)
    matrix_to_return = Matrix{UInt64}(undef, 2053, count_size(key)) #TOREVISE
    for i=1:length(D)
        for j=1:length(D[i])
            for vec in D[i]
                matrix_to_return = hcat(matrix_to_return,vec)
            end
        end
    end
    return matrix_to_return
end

# Create a non-overlapping vector of ranges
function create_range(x::Vector{Int64}) 
    vector_to_ret = Vector{UnitRange{Int64}}()
    start = 1
    for i=1:length(x)
        push!(vector_to_ret, start:start+x[i]-1)
        start +=x[i]
    end
    return vector_to_ret
end

bag_path = create_bags(dict_hashed["path"], "path")
bag_query = create_bags(dict_hashed["query"], "query")
bag_fragment = create_bags(dict_hashed["fragment"], "fragment")

bag_path_index = create_range(bag_indx_parts["path"])
bag_query_index = create_range(bag_indx_parts["query"])
bag_fragment_index = create_range(bag_indx_parts["fragment"])

# Vector of how many parts is the URL consisting of 
# ??
function bag_ind_of_bag()
    vec_to_ret = Vector{Int64}()
    for i=1:length(bag_indx_parts["path"])
        if bag_indx_parts["path"][i] !=0 && bag_indx_parts["query"][i] !=0
            push!(vec_to_ret, 2)
        else
            push!(vec_to_ret, 1)
        end
    end
    return vec_to_ret
end

# Waste?
#bag_bag_indx = create_range(bag_ind_of_bag())