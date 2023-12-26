
"""
variance covariance objects
"""

Base.@kwdef mutable struct GMMBootFits
    boot_fits                   # vector of all GMMFit objects
    boot_weights::Matrix        # nboot x n_obs matrix with bootstrap weights
    boot_fits_df::DataFrame     # df with all iterations for all bootstrap runs
end

# extent the "copy" method to the GMMOptions type
# Base.copy(x::GMMBootFits) = GMMBootFits([getfield(x, k) for k ∈ fieldnames(GMMBootFits)]...)

Base.@kwdef mutable struct GMMvcov
    method::Symbol
    V  = nothing
    ses = nothing
    W = nothing
    J = nothing
    Σ = nothing
    boot_fits::Union{GMMBootFits, Nothing} = nothing
    boot_fits_dict = nothing
end

### Asymptotic variance-covariance matrix

function jacobian(data, mom_fn::Function, myfit::GMMFit)
    
    moment_loaded = theta -> mean(mom_fn(data, theta), dims=1)

    try
        # try automatic differentiation
        return ForwardDiff.jacobian(moment_loaded, myfit.theta_hat)
    catch e
        throw(e)
        # fall back on finite differences
        println("jacobian: AD failed, falling back on finite differences")
        return FiniteDiff.finite_difference_jacobian(moment_loaded, myfit.theta_hat)
    end

end

function vcov_simple(data, mom_fn::Function, myfit::GMMFit)
    
    # jacobian
    J = jacobian(data, mom_fn, myfit)
   
    # weighting matrix
    W = myfit.W
    
    # simple, general: (JWJ')⁻¹ * JWΣW'J * (JWJ')⁻¹

    # estimate Σ
    m = mom_fn(data, myfit.theta_hat)
    N = myfit.n_obs
    Σ = Transpose(m) * m / N
    # display(Σ)

    invJWJ = inv(J * W * J')
    
    V = invJWJ * J * W * Σ * W' * J * invJWJ ./ N

    myfit.vcov = GMMvcov(
        method = :simple,
        V = V,
        J = J,
        W = W,
        Σ = Σ,
        ses = sqrt.(diag(V)))

    return
end

### Bayesian bootstrap

function boot_table(boot_fits::Vector{GMMFit})

    nboot = length(boot_fits)
    nparams = length(boot_fits[1].theta_hat)

    theta_hat_table = Matrix{Float64}(undef, nboot, nparams)
    for i=1:nboot
        theta_hat_table[i, :] = boot_fits[i].theta_hat
    end

    return theta_hat_table
end

function process_boot_fits(boot_fits::Vector{GMMFit})

    ### matrix with weights (row = bootstrap run, column = observation)
    all_boot_weights = hcat([boot_fits[i].weights for i=1:length(boot_fits)]...)
    for i=1:length(boot_fits)
        boot_fits[i].weights = nothing
    end

    ### full table of theta_hat's
    all_boot_fits = []
    nboot = length(boot_fits)
    for i=1:nboot
        temp_df = copy(boot_fits[i].all_model_fits)
        temp_df[!, :boot_idx] .= i
        push!(all_boot_fits, temp_df)

        boot_fits[i].all_model_fits = nothing
    end
    all_boot_fits = vcat(all_boot_fits...)

    # only keep the optimal iterations
    # mysample = all_boot_fits.is_optimum .== 1
    # all_fits = copy(all_boot_fits[mysample, :])

    return GMMBootFits(
        boot_fits = boot_fits,          # vector of GMMFit objects
        boot_weights = all_boot_weights,# nboot x n_obs matrix with bootstrap weights
        boot_fits_df = all_boot_fits    # df with all iterations for all bootstrap runs
    )
end






"""
I.i.d observations. Draw independent weights from a Dirichlet distribution with parameter 1.0. Assuming `data` is a DataFrame.
"""
function boot_weights_simple(rng, data, n_obs)
    if isa(data, DataFrame) 
        return rand(rng, Dirichlet(nrow(data), 1.0))        
    else
        @warn "`data` is not a DataFrame, using n_obs directly"
        return rand(rng, Dirichlet(n_obs, 1.0))
    end
    
end

function boot_weights_cluster(rng, idx_cluster_crosswalk, n_clusters)
    cluster_level_weights = rand(rng, Dirichlet(n_clusters, 1.0))

    # one step "join" to get the weight for the appropriate cluster
    return cluster_level_weights[idx_cluster_crosswalk]
end


function vcov_bboot(
    data, 
    mom_fn::Function, 
    theta0, 
    myfit::GMMFit; 
    W=I, 
    boot_weights::Union{Function, Symbol}=:simple, # accepts :simple, :cluster, or a user-provided function(rng, data, n_obs)
    nboot=100, 
    rng_initial_seed=1234,
    cluster_var=nothing, 
    run_parallel=true,
    opts::GMMOptions=default_gmm_opts())

    # copy options so we can modify them (trace and path)
    opts = deepcopy(opts)

    if !isnothing(cluster_var) && (boot_weights != :cluster)
        @error "cluster_var is specified but boot_weights is not :cluster. Proceeding without clustering."
    end

    # create path for saving results
    if (opts.path != "") && (opts.path != "")
        (opts.trace > 0) && println("creating path for saving results")
        bootpath = opts.path * "__boot__/"
        isdir(bootpath) || mkdir(bootpath)
    end

     ### bootstrap moment function 
    # need recenter so that it equal zero at theta_hat
    # see Hall and Horowitz (1996) for details
    
    # ! maybe load directly from fit
    m = mom_fn(data, myfit.theta_hat)
    
    mom_fn_boot = (data, theta) -> mom_fn(data, theta) .- mean(m, dims=1)

    n_obs = myfit.n_obs
    n_moms = myfit.n_moms

    # random number generator
    main_rng = MersenneTwister(rng_initial_seed)

    # generate bayesian bootstrap weight vectors (one per bootstrap run)
    if boot_weights == :simple
        boot_weights_fn = boot_weights_simple

    elseif boot_weights == :cluster
        # prep cluster

        @assert !isnothing(cluster_var) "cluster_var must be specified if boot_weights=:cluster"

        cluster_values = unique(data[:, cluster_var])

        ### ___idx_cluster___ has the index in the cluster_values vector of this row's value        
        # join to original data (must be a DataFrame)
        temp_df = DataFrame(string(cluster_var) => cluster_values, "___idx_cluster___" => 1:length(cluster_values))
        crosswalk_df = leftjoin(data[:,[cluster_var]], temp_df, on=cluster_var)

        # draw random weights for each cluster and fill in a vector as large as the original data
        boot_weights_fn = (rng, data, n_obs) -> boot_weights_cluster(rng, crosswalk_df.___idx_cluster___, length(cluster_values))
    else
        @assert isa(boot_weights, Function) "boot_weights must be :simple, :cluster, or a function(rng, data)"
        boot_weights_fn = boot_weights
    end

    all_boot_weights = []
    for i=1:nboot
        (opts.trace > 0) && println("generating bootstrap weights for run ", i)

        # the output is a vector of weights same size as the number of rows from the moment
        boot_weights = boot_weights_fn(main_rng, data, n_obs)

        # normalizing weights is important for numerical precision
        boot_weights ./= mean(boot_weights)

        push!(all_boot_weights, boot_weights)
    end

    # run bootstrap (serial or parallel)
    if !run_parallel
        all_boot_fits = Vector{GMMFit}(undef, nboot)
        for i=1:nboot
            all_boot_fits[i] = bboot(
                i, 
                data, 
                mom_fn_boot, # using the boot mom function (zero at theta_hat)
                theta0,
                all_boot_weights[i],
                W=W,
                opts=opts)
        end
    else
        # all_boot_fits = @showprogress pmap( 
        all_boot_fits = pmap( 
            i -> bboot(i, 
                    data, 
                    mom_fn_boot, # using the boot mom function (zero at theta_hat)
                    theta0,
                    all_boot_weights[i],
                    W=W,
                    opts=opts), 
            1:nboot)
    end

    # collect and process all bootstrap results
    boot_fits = process_boot_fits(all_boot_fits)

    # store results
    myfit.vcov = GMMvcov(
        method = :bayesian_bootstrap,
        boot_fits = boot_fits,
        V = cov(boot_table(boot_fits.boot_fits))
    )

    # save results to file?
    (opts.path != "") && write(myfit.vcov, opts)

    # delete all intermediate files with individual iteration results
    if opts.clean_iter 
        try
            (opts.trace > 0) && print("Deleting individual boot files and __boot__ subfolder...")
            rm(opts.path * "__boot__/", force=true, recursive=true)
            (opts.trace > 0) && println(" Done.")
        catch e
            @warn "Could not delete individual boot files and __boot__ subfolder."
        end
    end

    return boot_fits
end

function bboot(
    idx::Int64, 
    data, 
    mom_fn::Function,
    theta0,
    boot_weights;
    W=I,
    opts::GMMOptions)

    (opts.trace > 0) && println("bootstrap run ", idx)

    # path for saving results
    new_opts = deepcopy(opts)
    (new_opts.path != "") && (new_opts.path *= "__boot__/boot_" * string(idx) * "_")
    new_opts.trace = 0

    return fit(data, mom_fn, theta0, W=W, weights=boot_weights, run_parallel=false, opts=new_opts)
end

