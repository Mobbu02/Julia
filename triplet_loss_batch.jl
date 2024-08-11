using Revise
using Statistics, LinearAlgebra, Mill, Flux, Random, Distances, JLD2, HTTP, Plots, DelimitedFiles, OneHotArrays, ROC
using TSne, Clustering, MultivariateStats

####################################### Stragegies for mining #################################################################

# Calculate the hardest negative and hardest positive = hard batch
function hard_batch(batch::Matrix{Float32}, labels::Vector{Int64}, margin::Float64)::Float64
    # Check if all samples are of the same class
    if all(labels.==0) || all(labels .== 1)
        return 0.0
    end

    # Construct a distance matrix
    distances = pairwise(Euclidean(), batch, dims =2)

   
    # Positives
    positive_mask = get_positive_mask(labels)  # Get positive mask
    positive_distances = distances .* positive_mask # Compute allowed positives
    hardest_positives = maximum(positive_distances, dims=2) # batch_size x 1 matrix of maximums in each row

    # Negatives
    negative_mask = get_negative_mask(labels) # Get negative mask 
    max_achor_neg_dist = maximum(distances, dims=2) # Find maximums along each row
    anchor_neg_dist = distances .+ max_achor_neg_dist .* (1 .- negative_mask) # Add maximum to parts that cannot create a negative
    hardest_negatives = minimum(anchor_neg_dist, dims=2) 

    # Count the triplet loss
    triplet_loss = max.(hardest_positives-hardest_negatives .+ margin, 0.0)
    return mean(triplet_loss)
end

# Calculate hardest negative and mean positive
function hard_neg_mean_pos(batch::Matrix{Float32}, labels::Vector{Int64}, margin::Float64)::Float64
    # Check if all samples are of the same class
    if all(labels.==0) || all(labels .== 1)
        return 0.0
    end

    # Construct a distance matrix
    distances = pairwise(Euclidean(), batch, dims=2)
    
    # Positives
    positive_distances = positive_distance_mat(distances, labels)
    mean_positive = mean(positive_distances, dims=2)

    # Negatives
    negative_mask = get_negative_mask(labels) # Get negative mask 
    max_achor_neg_dist = maximum(distances, dims=2) # Find maximums along each row
    anchor_neg_dist = distances .+ max_achor_neg_dist .* (1 .- negative_mask) # Add maximum to parts that cannot create a negative
    hardest_negatives = minimum(anchor_neg_dist, dims=2) 

    # Count the triplet loss
    triplet_loss = max.(mean_positive - hardest_negatives .+ margin, 0)
    return triplet_loss
end

# Easy positive mining
function easy_pos_hard_neg_batch(batch::Matrix{Float32}, labels::Vector{Int64}, margin::Float64)::Float64
    # Check if all samples are of the same class
    if all(labels.==0) || all(labels .== 1)
        return 0.0
    end
    
    # Construct a distance matrix
    distances = pairwise(Euclidean(), batch, dims=2)

    # Easy positives
    positive_distances = positive_distance_mat(distances, labels)
    easy_positives = minimum(positive_distances, dims=2)

    # Negatives
    negative_distances = negative_distance_mat(distances, labels)
    hardest_negatives = minimum(negative_distances, dims=2)
    
    # Compute the triplet loss
    triplet_loss = max.(easy_positives - hardest_negatives .+ margin, 0.0)
    return mean(triplet_loss)
end

# Calculate semihard negative and hardest positive
function semi_hard_negative_batch(batch::Matrix{Float32}, labels::Vector{Int64}, margin::Float64)::Float64
    # Check if all samples are of the same class
    if all(labels.==0) || all(labels .== 1)
        return 0.0
    end

    # Construct a distance matrix
    distances = pairwise(Euclidean(), batch, dims=2)

    # Positives
    positive_distances = positive_distance_mat(distances, labels)
    hardest_positive = maximum(positive_distances, dims = 2)

    # Semi-hard negatives
    negative_mask = get_negative_mask(labels)
    only_neg = (distances .* negative_mask) .- hardest_positive
    pos_mask = only_neg .> 0
    semi_hard = minimum(only_neg .* pos_mask, dims=2)


    # Compute the triplet loss
    triplet_loss = max.(hardest_positive - semi_hard .+ margin, 0)
    return mean(triplet_loss)
end

function semi_hard_neg_mean_pos(batch::Matrix{Float32}, labels::Vector{Int64}, margin::Float64)::Float64
    # Check if all samples are of the same class
    if all(labels.==0) || all(labels .== 1)
        return 0.0
    end

    # Construct a distance matrix
    distances = pairwise(Euclidean(), batch, dims=2)

    # Positives
    positive_distances = positive_distance_mat(distances, labels)
    mean_positive = mean(positive_distances, dims=2)

    # Semi-hard negatives
    negative_mask = get_negative_mask(labels)
    only_neg = (distances .* negative_mask) .- mean_positive
    pos_mask = only_neg .> 0
    semi_hard = minimum(only_neg .* pos_mask, dims=2)

    triplet_loss = max.(mean_positive - semi_hard .+ margin, 0)
    return mean(triplet_loss)
end


# Get a positive mask
function get_positive_mask(labels::Vector{Int64})::Matrix{Float32}
    batch_size = length(labels) # Length of the batch
    ind_eq = I(batch_size) # Identity of batch_size x batch_size
    unq_ind = .!ind_eq  # Invert 
    label_eq = labels .== labels' # Check where the labels are equal
    mask = unq_ind .& label_eq
    return Float32.(mask)
end

# Get a negative mask
function get_negative_mask(labels::Vector{Int64})::Matrix{Float32}
    eq_lab = labels .== labels' # Calculate the mask, where the labels are equal 
    return .!eq_lab # Distinct labels
end

# Matrix of distances for positives
function positive_distance_mat(distances::Matrix{Float32}, labels::Vector{Int64})::Matrix{Float64}
    positive_mask = get_positive_mask(labels)
    positive_distances = distances .* positive_mask
    return positive_distances
end

# Matrix of distances for negatives
function negative_distance_mat(distances::Matrix{Float32}, labels::Vector{Int64})::Matrix{Float64}
    negative_mask = get_negative_mask(labels)
    max_achor_neg_dist = maximum(distances, dims=2)
    anchor_neg_dist = distances .+ max_achor_neg_dist .* (1 .- negative_mask) # Add maximum to parts that cannot create a negative
    return anchor_neg_dist
end

#################################################################


#################################################################

# Takes in a URL as string and divides it, then creates bagnodes and finally a single product node for single URL
function url_to_mill(input::String, ngrams::Int64, prime::Int64, base::Int64)::ProductNode
    url = HTTP.URI(input) # Create "URL" type
    
    # Create parts of the URL
    path = url.path
    query = url.query

    # Prepare input into functions
    path = filter(!isempty, String.(split(path,"/")[2:end]))
    path = isempty(path) ? [""] : path
    query = String.(split(query,"&"))

    # Construct the structure of the model node
    node = Mill.ProductNode((transform_url(path, ngrams, prime, base), 
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

# Pre-create a model, state of the saved model is loaded into this one
function precreate_model()::ProductModel
    model = ProductModel(tuple(
        BagModel(ArrayModel(Flux.Chain(Dense(2053 => 1024, bias = true, gelu), BatchNorm(1024), Dropout(0.3), Dense(1024=> 512, bias=true, gelu), BatchNorm(512), Dropout(0.25), Dense(512=>256, gelu) ) ),  #
                AggregationStack(SegmentedMean(256), SegmentedLSE(256))),

                BagModel(ArrayModel(Flux.Chain(Dense(2053 => 1024, bias = true, gelu ), BatchNorm(1024), Dropout(0.3), Dense(1024=> 512, bias=true, gelu), BatchNorm(512), Dropout(0.25), Dense(512=>256, gelu) ) ),  #
                AggregationStack(SegmentedMean(256), SegmentedLSE(256)))

    ), Flux.Chain(Dense(1024=> 512, bias = true, gelu),
        BatchNorm(512),
        Dropout(0.2),
        Dense(512=>256, bias=true, gelu),
        BatchNorm(256),
        Dropout(0.2),
        Dense(256=>128, bias = true, gelu),
        BatchNorm(128),
        Dense(128=> 2, bias=true)))

    return model
end

# Load the state of a model
function load_state!(model::ProductModel, path::String)::Nothing
    # Load the state of the model into the pre-created model 
    Flux.loadmodel!(model, JLD2.load(path, "model_state"))
    return nothing
end

# Function to create batches from grouped URLs, used while training the Triplet network
function create_batches(urls::Vector{String}, labels::Vector{Int64}, batch_size::Int64, ngrams::Int64, prime::Int64, base::Int64)
    # Pre-process URLs into ProductNodes and group by domain
    domain_dict = Dict{String, Vector{Tuple{ProductNode, Int64}}}()
    for (url, label) in zip(urls, labels) # Iterate through
        domain = HTTP.URI(url).host 
        node = url_to_mill(url, ngrams, prime, base) # Transfrom to MIL
        if haskey(domain_dict, domain)
            push!(domain_dict[domain], (node, label))
        else
            domain_dict[domain] = [(node, label)]
        end
    end

    # Function to shuffle and create batches
    function shuff_batch() 
        # Shuffle the domains
        domains = collect(keys(domain_dict))
        Random.shuffle!(domains)
        
        # Create batches
        batches = Vector{Tuple{Vector{ProductNode}, Vector{Int64}}}()
        current_batch_nodes = Vector{ProductNode}()
        current_batch_labels = Vector{Int64}()

        # Iterate through domains
        for domain in domains 
            domain_pairs = domain_dict[domain]
            for (node, label) in domain_pairs # Fill all instances of the domain in the batch
                push!(current_batch_nodes, node)
                push!(current_batch_labels, label)
            end

            # Push into the batch until it reaches batch size
            if length(current_batch_nodes) >= batch_size # Check if batch is biggger than batchsize, if yes, go to another batch
                push!(batches, (current_batch_nodes, current_batch_labels))
                current_batch_nodes = Vector{ProductNode}()
                current_batch_labels = Vector{Int64}()
            end
        end

        # Add any remaining nodes to the last batch
        if !isempty(current_batch_nodes)
            push!(batches, (current_batch_nodes, current_batch_labels))
        end

        # Shuffle batches to keep it random
        Random.shuffle!(batches)
        
        return batches
    end
    
    return shuff_batch # Return a function that will create batches every time
end

# Training the siamese network, return forward pass on evaluation data
function train_triplet_network!(model::ProductModel, pca_model::PCA{Float32}, train_transform::Vector{<:ProductNode}, training_urls::Vector{String}, training_labels::Vector{Int64}, evaluation_urls::Vector{<:ProductNode}, 
     evaluation_labels::Vector{Int64}, validation_urls::Vector{String}, validation_labels::Vector{Int64},
     batch_size::Int64, learning_rate::Float64, epochs::Int64, margin::Float64, path::String, numb_clusters::Int64, ngrams::Int64, base::Int64)::Nothing
    
    # Create the instances of the iterating functions
    train_batches = create_batches(training_urls, training_labels, batch_size, ngrams, 2053, base)
    validation_batches = create_batches(validation_urls, validation_labels, batch_size, ngrams, 2053, base)


    # Set up the optimizer
    opt = Flux.Optimiser(Adam(learning_rate))
    opt_state = Flux.setup(opt, model)

   
    # Create paths for saving 
    path_ongoing_train = joinpath(path, "train_during_training")
    if !isdir(path_ongoing_train)
        mkdir(path_ongoing_train)
    end
    path_ongoing_eval = joinpath(path, "eval_during_training")
    if !isdir(path_ongoing_eval)
        mkdir(path_ongoing_eval)
    end


    # Early stopping
    state = Dict("best_loss" => Inf, "no_improv" => 0, "patience"=> 3)
    
    function early_stop_cb(i::Int64)::Tuple{Bool, Float32}
        testmode!(model) # Set the model into testmode
        valid_loss = mean(hard_batch(model(batch[1]), batch[2], margin) for batch in validation_batches())
        @info "Validation loss is: $valid_loss"
        if valid_loss < state["best_loss"]
            state["best_loss"] = valid_loss
            state["no_improv"] = 0

            # Save the best model
            println("Best model saved at epoch $i.")
            s = Flux.state(model)
            jldsave(path*"/best_model.jld2", model_state=s);
        else
            state["no_improv"] +=1
        end      
        trainmode!(model) # Set the model back into trainmode
        return (state["no_improv"] > state["patience"], valid_loss)
    end

    # Save epochs
    epoch_losses = Float32[]

    trainmode!(model) # Set the model into a trainmode

    # Training loop
    for i in 1:epochs
        my_training_loader = train_batches()
        # Batch training
        for (urls, labels) in my_training_loader
            val, grads = Flux.withgradient(model) do m
                result = m(urls) # Calculate the forward pass on the current batch of URLs
                #batch_loss = hard_batch(result, labels, margin) # Calculate the triplet loss
                #batch_loss = hard_neg_mean_pos(result, labels, margin)
                #batch_loss = easy_pos_hard_neg_batch(result, labels, margin)
                #batch_loss = semi_hard_negative_batch(result, labels, margin)
                batch_loss = semi_hard_neg_mean_pos(result, labels, margin)
                return batch_loss
            end
            Flux.update!(opt_state, model, grads[1]) # Update the parameters of the model 
        end
        #@info "Epoch $i: Average Loss = $avg_epoch_loss"

        if i%2 == 0
            testmode!(model)
            cluster_kmeans(pca_model, model(evaluation_urls), evaluation_labels, numb_clusters, "evaluation", "after_epoch_$i", path_ongoing_eval*"/");
            cluster_kmeans(pca_model, model(train_transform), training_labels, numb_clusters, "training", "after_epoch_$i", path_ongoing_train*"/");
            trainmode!(model)
        end

        # Early stop + log of validation loss
        check, val_loss = early_stop_cb(i)
        push!(epoch_losses, val_loss)

        # If patience is exceeded, stop the training
        if check
            @info "Early stopping at epoch $i. Restoring best state."
            break
        end

    end
    
    # Plot the loss on the validation data and save
    pl = plot(1:length(epoch_losses), epoch_losses, title="Validation loss across epochs", xlabel="Epochs", ylabel="Loss");
    savefig(pl, path*"loss.svg");

    
    # Load the best performing model into the current one
    load_state!(model, path*"best_model.jld2") # Load the state of the save model into the pre-created one
    testmode!(model)

    # Evaluation the best model
    cluster_kmeans(pca_model, model(evaluation_urls), evaluation_labels, numb_clusters, "evaluation", "best", path_ongoing_eval*"/");
    cluster_kmeans(pca_model, model(train_transform), training_labels, numb_clusters, "training", "best", path_ongoing_train*"/");
    cluster_kmeans(pca_model, model(url_to_mill.(validation_urls, ngrams, 2053, base)), validation_labels, numb_clusters, "validation", "best", path_ongoing_train*"/");

    return nothing
end

# Return the model without the last reduction layer 
function get_part_of_network(model::ProductModel)::Tuple{ProductModel, Dense}    
    chain = model.m[1:end-1] # Get the chain part of the Product model
    sub_model = ProductModel(model.ms, chain) # Get the hierarchical structure for paths/queries
    #printtree(part_mod)
    #printtree(mod)
    return sub_model, model.m[end]
end


#################################################################
### Domain similarity
function dom_sim(urls::Vector{String})::Dict{String, Vector{Int64}}
    
    # Get the domains of the urls
    domains = [HTTP.URI(i).host for i in urls]

    # Get domains and their indexes
    domain_dict = Dict{String, Vector{Int64}}()
    for (index, domain) in enumerate(domains)
        
        if haskey(domain_dict, domain)
            push!(domain_dict[domain], index)
        else
            domain_dict[domain] = [index]
        end
    end
   

    # Select only duplicates
    dupes = Dict{String, Vector{Int64}}()
    for (domain, indexes) in domain_dict
        if length(indexes) > 1
            dupes[domain] = indexes
        end
    end

    return dupes
end

# Plot groups of points 
function plot_dom_sim_pca(data::Matrix{Float32}, idx_dupes::Vector{Vector{Int64}}, path::String)::Nothing
    colorm = distinguishable_colors(length(idx_dupes))
    
    domain_colors = Dict{Int64, RGB}()
    for (i, idxs) in enumerate(idx_dupes)
        for idx in idxs
            domain_colors[idx] = colorm[i]
        end
    end

    # Plot in groups of 5 duplicates
    batch_size = 5
    max_x = maximum(data[1,:]) +0.5
    min_x = minimum(data[1,:]) -0.5
    max_y = maximum(data[2,:]) +0.5
    min_y = minimum(data[2,:]) -0.5
    num_batches = ceil(Int, length(idx_dupes) / batch_size)
    numb = 1
    if !isdir(path)
        mkdir(path)
    end
    
    for batch in 1:num_batches
        start_idx = (batch - 1) * batch_size + 1
        end_idx = min(batch * batch_size, length(idx_dupes))
        
        # Collect batch indices
        batch_idxs = reduce(vcat, idx_dupes[start_idx:end_idx])
        
        batch_xs = data[1, batch_idxs]
        batch_ys = data[2, batch_idxs]
        batch_colors = [domain_colors[idx] for idx in batch_idxs]

        a = scatter(batch_xs, batch_ys, color = batch_colors, legend = false, markersize = 10, xlims=(min_x, max_x), ylims=(min_y, max_y), size=(1000,1000));

        savefig(a, path*"/$numb.svg");

        numb +=1
    end

    return nothing
end

# Cluster the 128 dimensional embeddings using Kmeans
function cluster_embeddings_with_domains(pca_mod::PCA{Float32}, urls::Matrix{Float32}, idx_dupes::Vector{Vector{Int64}}, path::String, name::String, state::String)::Float64

    number_of_clusters = length(idx_dupes) # Number of duplicate domains
    idxs = vcat(idx_dupes...) # Unpack into a single vector
    dupes_vecs = urls[:, idxs] # Choose only the duplicate vectors

    max_retries = 20
    retries = 0
    kmeans_res = nothing
    clus_assignments = nothing

    while retries <= max_retries
        try
            # Clustering with kmeans
            kmeans_res = kmeans(dupes_vecs, number_of_clusters,  display =:none)
            clus_assignments = kmeans_res.assignments # Assignments to clusters
            break # Exit the loop if kmeans is successful
        catch e
            println("K-means failed: ", e)
            retries += 1
            if retries == max_retries
                println("Maximum retries reached for K-means.")
                save_error_state(dupes_vecs, idx_dupes, path, "kmeans_failure")
                return -1.0  # Indicate failure with a negative value
            end
        end
    end

    # Create labels based on idx_dupes
    labels = Vector{Int64}()
    for (i, idx_group) in enumerate(idx_dupes)
        append!(labels, fill(i, length(idx_group)))
    end

    # Calculate the V measure
    v_measure = Clustering.vmeasure(clus_assignments, labels)
    #println("V-measure: ", v_measure)

    # PCA
    reduced_vecs = MultivariateStats.transform(pca_mod, dupes_vecs)

    jldsave(path*"/$name"*"_"*state*"_points.jld2", s=reduced_vecs)

    #pl = scatter(reduced_vecs[1,:], reduced_vecs[2,:], group=clus_assignments, legend=false, markersize=5, title=name);
    #savefig(pl, path*"/$name"*"_"*state*"_"*"similar.svg");

    #println(clustering_goodness(urls, clus_assignments, idx_dupes))

    return v_measure
end

# Count silhouette coefficient based on domains
function count_sil_coef_dups(pca_mod::PCA{Float32}, matrix::Matrix{Float32}, indexes::Vector{Vector{Int64}}, path::String, state::String)::Float64
    idxs = vcat(indexes...) # Unpack into a single vector
    dupes_vecs = matrix[:, idxs] # Choose only the duplicate vectors

    
    # Create labels based on indexes
    labels = Vector{Int64}()
    for (i, idx_group) in enumerate(indexes)
        append!(labels, fill(i, length(idx_group)))
    end


    # Compute the silhouette coeficient
    dist_mat = pairwise(Euclidean(), dupes_vecs, dims=2) # Distance matrix between the columns
    silh_cof = silhouettes(labels, dist_mat) # Silhouette coeficient


    # Additional PCA
    data = MultivariateStats.transform(pca_mod, dupes_vecs)
    a = scatter(data[1,:], data[2,:], group = labels);
    path_to_sil = joinpath(path, "silhouettes")
    if !isdir(path_to_sil)
        mkdir(path_to_sil)
    end
    
    jldsave(path_to_sil*"/"*"domains_together_"*state*".jld2", dat = data)
    jldsave(path_to_sil*"/"*"domains_together_labels_"*state*".jld2", lab = labels)
    savefig(a, path_to_sil*"/"*state*"_domains.svg");

    return mean(silh_cof)
end

#################################################################






#################################################################
#### Clustering ####

# MinMax scaling of data
function minmax_scale(data::Matrix{Float32})
    min_val = minimum(data, dims=1)
    max_val = maximum(data, dims=1)
    return (data .- min_val) ./ (max_val .- min_val)
end

# Inertia computation for elbow method
function compute_inertia(data::Matrix{Float32}, centers::Matrix{Float32}, assignments::Vector{Int})::Float64
    inertia = 0.0
    for i in 1:size(data, 2)
        cluster_center = centers[:, assignments[i]]
        inertia += sum((data[:, i] .- cluster_center) .^ 2)
    end
    return inertia
end

# Elbow method for optimal number of clusters
function elbow(urls::Matrix{Float32}, k::Int64)::Int64
    distortions = zeros(k)
    for j in 1:k
        result = kmeans(urls, j, maxiter=1000, display =:none);
        distortions[j] = compute_inertia(urls, result.centers, assignments(result))
    end
    #p = plot(1:k, distortions, xlabel="Number of clusters", ylabel="Distortions", marker=:o)
    #display(p)
    return argmin(diff(diff(distortions))) + 1
end

# Kmeans clustering on input data
function cluster_kmeans(pca_model::PCA{Float32}, inc_data::Matrix{Float32}, labels::Vector{Int64}, n_clusters::Int64, name::String, state::String, path::String)::Plots.Plot{Plots.GRBackend}
    try
        # 'Normalize' the data
        urls = minmax_scale(inc_data)

        ##################################################################
        # Find optimal number of clusters
        k = 2 #elbow(urls, n_clusters)

        #println("Optimal number of clusters by the elbow method is: $k.")
        user_k = k

        # Retry mechanism for K-means clustering
        result = nothing
        max_retries = 20
        retries = 0

        # Try multiple times
        while retries < max_retries
            try
                result = kmeans(urls, user_k, maxiter=700, display =:none)
                break  # Exit the loop if kmeans is successful
            catch e
                println("K-means failed: ", e) # log a message
                retries += 1
                if retries == max_retries
                    println("Maximum retries reached for K-means.")
                    save_error_state(inc_data, labels, path, "kmeans_failure")
                    return nothing  # Skip if failed
                end
            end
        end
        assignment = assignments(result) 

        # Saving path
        sil_path = path * "/silhouettes"
        if !isdir(sil_path)
            mkdir(sil_path)
        end

        # Get clustering quality index
        nclusters = 2:k+1
        clusterings = []
        try
            clusterings = kmeans.(Ref(urls), nclusters)
        catch e
            println("K-means clustering for quality measurement failed: ", e)
            save_error_state(inc_data, labels, path, "quality_measurement_failure")
        end

        quality_plot = plot(nclusters, clustering_quality.(Ref(urls), clusterings, quality_index=:silhouettes), marker=:circle, title="Silhouette",
                            xlabel="Clusters", ylabel="Value of coefficient")
        savefig(quality_plot, sil_path * "/" * name * "_" * state * "_silhouette.svg")
        ##################################################################

        # Reduce the dimension with PCA model
        data = transform_with_pca(pca_model, urls)
        
        # Prepare colors
        colors = palette(:tab10)
        cluster_colors = [colors[i] for i in assignment]

        # Plot the PCA-reduced data with cluster colors
        p_pca = catter(data[1, :], data[2, :], color=cluster_colors, title="K-means Clustering with PCA, $user_k clusters", xlabel="PC1", ylabel="PC2");

        # Prepare class plot
        c_color = ["blue", "red"]
        labels_desc = ["benign", "malicious"]
        p_class = scatter(xlabel="PC1", ylabel="PC2", size=(1200,1200));
        
        for i in 0:1
            scatter!(p_class, data[1, labels .== i], data[2, labels .== i], color=c_color[i+1], label=labels_desc[i+1],markersize=10, tickfontsize=30, guidefontsize=35, legendfontsize=25); 
        end


        # Save more plots
        savefig(p_class, path*name*"_"*state*"_dispersion.svg");
        b = scatter(data[1, labels .==0], data[2, labels .==0], color="blue");
        c = scatter(data[1, labels .==1], data[2, labels .==1], color="red");
        savefig(b, path*name*"_"*state*"_benign_alone.svg");
        savefig(c, path*name*"_"*state*"_malicious_alone.svg");

        # Final plot
        final_plot = plot(p_pca, p_class, layout=(2, 1))
        savefig(final_plot, path * name * "_" * state * "_final_plot.svg")
        return final_plot

    catch e # Catch an error if kmeans failed
        println("Error encountered in cluster_kmeans: ", e)
        save_error_state(inc_data, labels, path, "general_failure")
        return nothing
    end
end

# Function to save the state of the data when an error occurs
function save_error_state(inc_data::Matrix{Float32}, labels::Vector{Int64}, path::String, error_type::String)::Nothing
    error_path = joinpath(path, "error_states")
    if !isdir(error_path)
        mkdir(error_path)
    end
    filename = joinpath(error_path, "error_state_$(error_type).jld2")
    JLD2.jldsave(filename, compress=true, iotype=JLD2.MmapIO, inc_data=inc_data, labels=labels)
    println("Saved error state to: ", filename)
    return nothing
end
#################################################################


#################################################################

### Clasification

# Cold treshold for converting a vector of probabilites into predictions
function cold_treshold(probs::Vector{Float32}, treshold::Float64 = 0.5)::Vector{Int64}
    return map(x-> x < treshold ? 1 : 0, probs)
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

    # println("Confusion matrix: ", conf_matrix)
    # println("Precision:", precis)
    # println("Recall: ", recall)
    # println("Mathews correlation coefficient: ", MCC(conf_matrix))

    return Evaluation(round(recall, digits = 3), round(precis, digits = 3), conf_matrix[1,1], conf_matrix[1,2], conf_matrix[2,2], conf_matrix[2,1], f_score, round(mathew, digits=3))
end

# Count the confusion matrix from the predicted and true labels
function confusion_matrix(true_labels::Vector{Int64}, predicted_labels::Vector{Int64})::Matrix{Int64}
    tp = sum((true_labels .== 1) .& (predicted_labels .== 1))
    tn = sum((true_labels .== 0) .& (predicted_labels .== 0))
    fp = sum((true_labels .== 0) .& (predicted_labels .== 1))
    fn = sum((true_labels .== 1) .& (predicted_labels .== 0))
    return [tp fp; fn tn]
end


# Calculte the roc curve and pr curve, both are saved
function rocA(probs::Vector{Float32}, true_labels::Vector{Int64}, path::String, state::String)::Nothing
    treshold_range = 0.0:0.001:1.0    # Treshold range of values
    predicted_labels = map(x-> cold_treshold(probs, x), treshold_range)     # Vector of vectors of predicted labels for each treshold    
    tpr, fpr, prec = tpr_fpr(predicted_labels, true_labels)

    # Save path
    path_roc = joinpath(path, "roc_data")
    if !isdir(path_roc)
        mkdir(path_roc)
    end

    # Plot the ROC curve
    roc_curve = plot(fpr, tpr, label="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", legend=:bottomright);
    plot!(roc_curve,[0, 1], [0, 1], label="Random Classifier", linestyle=:dash);
    savefig(roc_curve, path_roc*"/"*state*"_roc.svg");


    # Plot the PR curve
    pr_curve = plot(tpr, prec, label="Precision/Recall Curve", xlabel="Recall", ylabel="Precision", legend=:bottomright, xlims=(0,1), ylims=(0,1));
    hline!(pr_curve, [0.5], label="Baseline Classifier", linestyle=:dash);
    savefig(pr_curve, path_roc*"/"*state*"_pr.svg");
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

# Phi coeficient ~ mathews correlation coefficient
function MCC(C::Matrix{Int64})::Float64
    #                [TP, FP]
    #                [FN, TN]
    return (C[1,1]*C[2,2] - C[1,2]*C[2,1])/(sqrt((C[1,1]+C[1,2])*(C[1,1]+C[2,1])*(C[2,2]+C[2,1])*(C[2,2]+C[1,2])))
end

#  Function calculates the accuracy from the predicted labels
function calculate_accuracy(predicted_labels::Vector{Int64}, true_labels::Vector{Int64})::Float64
    correct_predictions = sum(predicted_labels .== true_labels)
    total_predictions = length(true_labels)
    accuracy = correct_predictions / total_predictions
    return accuracy
end

## Retraining
function clasify(model::ProductModel, urls::Vector{<:ProductNode}, labels::Vector{Int64}, treshold::Float64, io::IOStream, path::String, state::String)::Nothing
    predictions = model(urls) # Predict
    probabilities = softmax(predictions)[1,1:end] # Probabilities
    predicted_labels = cold_treshold(probabilities, treshold) # Transform into labels
    evaluation = evaluate_performance(labels, predicted_labels) # Evaluate performce
    accuracy = calculate_accuracy(predicted_labels, labels)

    # ROC and save
    rocA(probabilities, labels,path,state)
    roc_curve = ROC.roc(probabilities, labels)
    path_to = joinpath(path, "roc_data")
    jldsave(path_to*"/"*state*"_roc_data.jld2", roc_c = roc_curve)
    area_under = ROC.AUC(roc_curve);
    jldsave(path_to*"/"*state*"_probabilities_labels.jld2", probs = probabilities, labs = labels)


    # Write results into a file
    println(io, state)
    println(io, "Precision is: ", evaluation.precision)
    println(io, "Recall is: ", evaluation.recall)
    println(io, "Fscore is: ", evaluation.F_score)
    println(io, "Accuracy is: ", accuracy)
    println(io, "Area under the curve: ", area_under)
    println(io, evaluation.true_pos, " , ", evaluation.false_pos, " , ", evaluation.false_neg, " , ", evaluation.true_neg)
    println(io, "")

    return nothing
end

# Train the classification model
function train_to_classify!(model::ProductModel, training_urls::Vector{<:ProductNode}, training_labels::Vector{Int64}, 
    validation_urls::Vector{<:ProductNode}, validation_labels::Vector{Int64}, learning_rate::Float64, batchsize::Int64, path::String, state_train::String, epochs::Int64)::Nothing

    # Create a loss function
    loss(x, y) = Flux.Losses.logitbinarycrossentropy(x, OneHotArrays.onehotbatch(y .+ 1, 1:2))  # Onehot inside from training labels

    # Create dataloaders for training and validation
    training_data_loader = Flux.DataLoader((training_urls, training_labels), batchsize = batchsize, shuffle = true, partial = false)
    valid_data_loader = Flux.DataLoader((validation_urls, validation_labels), batchsize = batchsize, shuffle = true, partial = false)

    # Early stopping
    # Stopping  criteria for training loop
    state = Dict("best_loss" => Inf, "no_improv" => 0, "patience"=> 5)

    opt = Flux.Optimiser(Adam(learning_rate))
    opt_state = Flux.setup(opt, model)

    if state_train == "after"
        # Freeze layers
        println("Freezing layers!")
        Flux.freeze!(opt_state.m.layers[1:end-1])
        Flux.freeze!(opt_state.ms)
    end

    # Early stopping callback
    function early_stop_cb()::Bool
        testmode!(model)
        valid_loss = mean(loss(model(batch[1]), batch[2]) for batch in valid_data_loader)
        #println("Validation loss is: ", valid_loss)
        if valid_loss < state["best_loss"]
            state["best_loss"] = valid_loss
            state["no_improv"] = 0

            # Save best model
            s = Flux.state(model)
            jldsave(path*"/"*state_train*"_best_classification_model.jld2", model_state=s);
        else
            state["no_improv"] += 1
        end
        trainmode!(model)
        return state["no_improv"] > state["patience"]
    end

    stopped_at = false  # Early stop signalizer

    ## Training phase
    trainmode!(model)

    # Training loop
    for i in 1:epochs
        #@info "Epoch number $i."
        is_finite = true # Used for checking if loss is finite

        # Batch work
        for (x_batch, y_batch) in training_data_loader

            # Calculate the loss and gradients
            val, grads = Flux.withgradient(model) do m
                result = m(x_batch)
                batch_loss = loss(result, y_batch)
                return batch_loss #+ reg_loss
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

        # # Early stopping
        if early_stop_cb()
           stopped_at = true
           break
        end

        # Control finite loss
        if !is_finite
            break
        end
        
    end

    load_state!(model, path*"/"*state_train*"_best_classification_model.jld2"); # Load the state of the save model into the pre-created one

    ## Testing phase
    testmode!(model) # Turn off dropout
    return nothing
end

#  Create the clasification model from the triplet network
function prepare_model(sub_model::ProductModel, final_layer::Dense)::ProductModel

    # Rebuild the classification model
    new_chain = Chain(sub_model.m..., final_layer) # Reconstruct the last network
    full_model = ProductModel(sub_model.ms, new_chain)
    return full_model
end

# Retrain the classifier after the Triplet Network
function retrain_classifier(sub_model::ProductModel, training_urls::Vector{<:ProductNode}, training_labels::Vector{Int64}, 
    validation_urls::Vector{<:ProductNode}, validation_labels::Vector{Int64}, evaluation_urls::Vector{<:ProductNode}, evaluation_labels::Vector{Int64},
    learning_rate::Float64, batchsize::Int64, treshold::Float64, final_layer::Dense, path::String)::ProductModel

    # Prepare the model
    model = prepare_model(sub_model, final_layer)

    # Train model
    train_to_classify!(model, training_urls, training_labels, validation_urls, validation_labels, learning_rate, batchsize, path, "after", 30)

    return model
end

#################################################################

# Fit PCA on training data
function fit_pca_on_training(training_data::Matrix{Float32}, maxoutdims::Int64=2)::PCA{Float32}
    
    # Fit a PCA model on the training data
    pca_model = fit(PCA, training_data; maxoutdim=maxoutdims)
    return pca_model
end

# Transform data with PCA
function transform_with_pca(pca_model::PCA{Float32}, data::Matrix{Float32})::Matrix{Float32}
    transformed_data = MultivariateStats.transform(pca_model, data)
    return transformed_data
end

# Perform the training of the Triplet network
function perform_training(sub_model::ProductModel, pca_model::PCA{Float32}, training_data::Vector{<:ProductNode}, training_urls::Vector{String}, training_labels::Vector{Int64}, evaluation_urls::Vector{<:ProductNode}, 
    evaluation_labels::Vector{Int64}, validation_urls::Vector{String}, validation_labels::Vector{Int64},
    batch_size::Int64, learning_rate::Float64, epochs::Int64, margin::Float64, numb_clusters::Int64, path::String, vecs::Vector{Vector{Int64}}, base::Int64, ngrams::Int64)::Tuple{Float64, Float64}

    ## Before training
    # Kmeans
    before_p1 = cluster_kmeans(pca_model, sub_model(training_data), training_labels, numb_clusters, "training", "before", path)
    processed_eval_urls_before = sub_model(evaluation_urls)
    before_p2 = cluster_kmeans(pca_model, processed_eval_urls_before, evaluation_labels, numb_clusters, "evaluation", "before", path)
    v_measure_before = cluster_embeddings_with_domains(pca_model, processed_eval_urls_before, vecs, path, "evaluation", "before")

    # Save kmeans plots
    before_final = plot(before_p1, before_p2, layout = (1,2), size=(1600,1200));
    savefig(before_final, path*"before_training_both.svg");

    ## Train the triplet network
    train_triplet_network!(sub_model, pca_model, training_data, training_urls, training_labels,evaluation_urls, evaluation_labels, validation_urls, validation_labels, batch_size, learning_rate, epochs, margin, path, numb_clusters, ngrams, base)

    ## After training
    # Kmeans 
    after_p1 = cluster_kmeans(pca_model, sub_model(training_data), training_labels, numb_clusters, "training", "after", path)
    processed_eval_urls_after = sub_model(evaluation_urls)
    after_p2 = cluster_kmeans(pca_model, processed_eval_urls_after, evaluation_labels, numb_clusters, "evaluation", "after", path)
    v_measure_after = cluster_embeddings_with_domains(pca_model, processed_eval_urls_after, vecs, path, "evaluation", "after")


    # Plot together
    after_final = plot(after_p1, after_p2, layout = (1,2), size = (1600,1200));
    savefig(after_final, path*"after_training_both.svg");

    return (v_measure_before, v_measure_after)
end 

# Function to unzip URLs and labels
function unzip(pairs::Vector{Tuple{String, Int64}})
    urls = String[]
    labels = Int64[]
    for (url, label) in pairs # push pairs - (url, label)
        push!(urls, url)
        push!(labels, label)
    end
    return urls, labels
end

# Function to organize URLs into training, validation, and evaluation sets
function split_urls(urls::Vector{String}, labels::Vector{Int64}, train_ratio::Float64, val_ratio::Float64)::Tuple{Vector{String}, Vector{Int64}, Vector{String}, Vector{Int64}, Vector{String}, Vector{Int64}}
    # Extract domains and group URLs by domain
    domain_dict = Dict{String, Vector{Tuple{String, Int64}}}()
    for (url, label) in zip(urls, labels)
        domain = HTTP.URI(url).host
        if haskey(domain_dict, domain)
            push!(domain_dict[domain], (url, label))
        else
            domain_dict[domain] = [(url, label)]
        end
    end

    # Shuffle and split domains
    domains = collect(keys(domain_dict))
    Random.shuffle!(domains)

    # Calculate the number of domains for each split
    total_domains = length(domains)
    
    num_train = Int(floor(train_ratio * total_domains))
    num_val = Int(floor(val_ratio * total_domains))

    # Split the domains
    train_domains = domains[1:num_train]
    val_domains = domains[num_train+1:num_train+num_val]
    eval_domains = domains[num_train+num_val+1:end]

    # Assign URLs and labels based on the domain split
    train_urls, train_labels = unzip(vcat([domain_dict[domain] for domain in train_domains]...))
    val_urls, val_labels = unzip(vcat([domain_dict[domain] for domain in val_domains]...))
    eval_urls, eval_labels = unzip(vcat([domain_dict[domain] for domain in eval_domains]...))

    return train_urls, train_labels, val_urls, val_labels, eval_urls, eval_labels
end

# Main
function main(urls_to_load::Int64, run_id::Int64)::Nothing
    train_ratio = 0.6
    val_ratio = 0.2
    println("RUN $run_id")
    try
        # Generate random number every time
        random_seed = 1328929158#rand(UInt32) # Seed is the same as was used in the training phase
        println(random_seed)
        #println("Seed: ", random_seed)
        new_indexing = randcycle(MersenneTwister(random_seed), 544708)     # Generate permutation of dataset length, seed the generator with random_seed
        idx_to_use = new_indexing[1:urls_to_load] # Select the urls to use from the random permutation

        # Read lines
        all_urls = JLD2.load("final_mix_urls.jld2", "all_urls")[idx_to_use]    #  Load URL addresses
        all_labels = JLD2.load("final_mix_labels.jld2", "all_labels")[idx_to_use]    # Load labels for URLs

        # Split the data
        training_lines, training_labels, valid_lines, valid_labels, eval_lines, eval_labels = split_urls(all_urls, all_labels, train_ratio, val_ratio)

        # Print the sizes of each split
        println("Number of training URLs: ", length(training_lines))
        println("Number of validation URLs: ", length(valid_lines))
        println("Number of evaluation URLs: ", length(eval_lines))

        # Transformation parameters
        prime = 2053
        base = 256 # 512
        ngrams = 3 # 2

        # Prepare training and evaluation
        training_urls = url_to_mill.(training_lines, ngrams, prime, base)
        eval_urls = url_to_mill.(eval_lines, ngrams, prime, base) 
        valid_urls = url_to_mill.(valid_lines, ngrams, prime, base)

        
        # Path to main directory
        base_path = "triplet_" * string(base) * "_" * string(ngrams)
        if !isdir(base_path)
            mkdir(base_path)
        end
        
        # Silhouette coefficient on pure domains
        url_dupes = collect(values(dom_sim(eval_lines))) # Vector of vectors, where each vector contains the indexes of duplicates for each domain
        
        # Define hyperparameter ranges
        margins = [5]#, 0.3, 0.5, 0.7, 0.9]
        batch_sizes = [64]#, 32]
        learning_rates = [0.005]# , 0.0005]
        epochs = 100
        numb_clusters = 7

        max_retries = 7
        # Perform the training with specific hyperparameters
        for margin in margins
            for batch_size in batch_sizes
                for learning_rate in learning_rates
                    retries = 0

                    # Construct paths
                    param_path = joinpath(base_path, string(margin) * "_" * string(batch_size) * "_" * string(learning_rate))
                    run_path = joinpath(param_path, "run_" * string(run_id))
                    if !isdir(param_path)
                        mkdir(param_path)
                    end
                    if !isdir(run_path)
                        mkdir(run_path)
                    end

                    # Try until successfull or exhaust the number of tries
                    while retries <= max_retries
                        try
                            println("trying for the $retries time!.")
                            println("margin: ", margin, ", batch size: ", batch_size, ", learning rate: ", learning_rate)

                            # Create the model 
                            model = precreate_model()

                            # Train the classification model
                            train_to_classify!(model, training_urls, training_labels, valid_urls, valid_labels, 0.003, 128, run_path, "before", 2) # Train the classification model

                            # Save results from the classification + seed
                            open(run_path * "/classification_results.txt", "w") do io
                                println(io, "Seed is: ", random_seed)
                                println(io, "")
                                clasify(model, eval_urls, eval_labels, 0.5, io, run_path * "/", "before")
                            end

                            # Load the best model
                            load_state!(model, run_path * "/before_best_classification_model.jld2") # Load the state of the saved model into the pre-created one
                            testmode!(model) # Test mode 
                            sub_model, final_layer = get_part_of_network(model) # Remove the last reduction layer of the binary classification model

                            # Fit PCA on training data
                            pca_mod = fit_pca_on_training(sub_model(training_urls))

                            # Compute silhouette coefficient before training
                            cof_before = count_sil_coef_dups(pca_mod, sub_model(eval_urls), url_dupes, run_path, "before")

                            # Triplet training
                            v_measures = perform_training(sub_model, pca_mod, training_urls, training_lines, training_labels, eval_urls, eval_labels, valid_lines, valid_labels, batch_size, learning_rate, epochs, margin, numb_clusters, run_path * "/", url_dupes, base, ngrams) 

                            # Compute silhouette coefficient after training
                            cof_after = count_sil_coef_dups(pca_mod, sub_model(eval_urls), url_dupes, run_path, "after")

                            # Retrain for classification
                            new_clas_model = retrain_classifier(sub_model, training_urls, training_labels, valid_urls, valid_labels, eval_urls, eval_labels, learning_rate, 128, 0.5, final_layer, run_path * "/")

                            # Write results of classification again
                            open(run_path * "/classification_results.txt", "a") do io
                                clasify(new_clas_model, eval_urls, eval_labels, 0.5, io, run_path * "/", "after")
                            end

                            # Write V measures into a file
                            open(run_path * "/classification_results.txt", "a") do io
                                println(io, "V-measure before training: ", v_measures[1])
                                println(io, "Silhouette coefficient before training: ", cof_before)
                                println(io, "")
                                println(io, "V-measure after training: ", v_measures[2])
                                println(io, "Silhouette coefficient after training: ", cof_after)
                            end
                            flush(stdout)
                            break  # Exit retry loop if successful

                        catch e
                            println("Error with parameters margin=$margin, batch_size=$batch_size, learning_rate=$learning_rate: ", e)
                            retries += 1
                            if retries == max_retries
                                println("Maximum retries reached for parameters margin=$margin, batch_size=$batch_size, learning_rate=$learning_rate.");
                                save_error_state(sub_model(training_urls), training_labels, run_path, "param_failure");
                            end
                        end
                    end
                end
            end
        end
    catch e
        println("Error encountered during run $run_id !")
        Base.show_backtrace(stderr, catch_backtrace())
    end

    println("Ending run $run_id")

    return nothing 
end


# Run main loop several times to perform robust training 
function run_main(numb_runs::Int64)::Nothing
    for run_id in 1:numb_runs
        try
            println("Starting run $run_id !")
            main(100000, run_id)
            println("Run completed!")
        catch e
            println("Run $run_id failed!")
            Base.show_backtrace(stderr, catch_backtrace())
        end
    end
    return nothing
end

#run_main(1)