
### Asymptotic variance-covariance matrix

function jacobian(data, mom_fn::Function, myfit::GMMFit)
    
    moment_loaded = theta -> mean(mom_fn(data, theta), dims=1)

    try
        # try automatic differentiation
        return ForwardDiff.jacobian(moment_loaded, myfit.theta_hat)
    catch
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
    N = size(m, 1)
    Σ = Transpose(m) * m / N
    # display(Σ)

    invJWJ = inv(J * W * J')
    
    V = invJWJ * J * W * Σ * W' * J * invJWJ ./ N

    myfit.vcov = Dict(
        :method => Vcov.simple(), #"plain",
        :V => V,
        :J => J,
        :W => W,
        :Σ => Σ,
        :N => N, # TODO : should really move this up to main fit object
        :ses => sqrt.(diag(V))
    )

    return
end

### Bayesian bootstrap

struct bayesian_bootstrap <: CovarianceEstimator
end

Base.@kwdef mutable struct GMMBootFits
    all_theta_hat  # nboot x nparams matrix with result for each bootstrap run
    all_model_fits # df with results (one per bootstrap run)
    all_boot_fits  # df with all iterations for all bootstrap runs
end

function Base.show(io::IO, r::GMMBootFits)
    println("Baysian bootstrap results")
    display(r.all_model_fits)
    display(r.all_boot_fits)
end

function write(boot_fits::GMMBootFits, opts::GMMOptions)
    if opts.path == ""
        return
    end

    # temp_df = DataFrame(boot_results.table, :auto)

    CSV.write(opts.path * "fits_boot.csv", boot_fits.all_model_fits)
    CSV.write(opts.path * "fits_boot_all.csv", boot_fits.all_boot_fits)
end

function process_boot_fits(lb::Vector{GMMFit})

    all_boot_fits = []
    nboot = length(lb)
    for i=1:nboot
        temp_df = copy(lb[i].all_fits)
        temp_df[!, :boot_idx] .= i
        push!(all_boot_fits, temp_df)
    end
    all_boot_fits = vcat(all_boot_fits...)

    mysample = all_boot_fits.is_optimum .== 1
    all_fits = copy(all_boot_fits[mysample, :])

    nparams = length(lb[1].theta_hat)
    # nboot = length(lb)

    fit_table = Matrix{Float64}(undef, nboot, nparams)
    for i=1:nboot
        fit_table[i, :] = lb[i].theta_hat
    end

    return GMMBootFits(
        all_theta_hat = fit_table,
        all_fits = all_fits,
        all_boot_fits = all_boot_fits
    )
end


function gen_boot_rngs(boot_n_runs, rng_initial_seed)
  
    # Random number generators (being extra careful) one per bootstrap run
    master_rng = MersenneTwister(rng_initial_seed)
    boot_rngs = Vector{Any}(undef, boot_n_runs)

    # each bootstrap run gets a different random seed
    # as we run the bootrap in separate rounds, large initial skip
    # boostrap_skip = (boot_round-1)*boot_n_runs + i
    for i=1:boot_n_runs
        println("creating random number generator for boot run ", i)
        boot_rngs[i] = randjump(master_rng, big(10)^20 * i)
    end

    return boot_rngs
end

"""
I.i.d observations. Draw independent weights from a Dirichlet distribution with parameter 1.0. Assuming `data` is a DataFrame.
"""
function boot_weights_simple(rng, data)
    @assert isa(data, DataFrame) "`data` must be a DataFrame"
    return rand(rng, Dirichlet(nrow(data), 1.0))
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
    boot_weights::Union{Function, Symbol}=:simple,
    nboot=100, 
    rng_initial_seed=1234,
    cluster_var=nothing, 
    run_parallel=true,
    opts::GMMOptions=default_gmm_opts())

    # create path for saving results
    if (opts.path != "") && (opts.path != "")
        (opts.trace > 0) && println("creating path for saving results")
        bootpath = opts.path * "__boot__/"
        isdir(bootpath) || mkdir(bootpath)
    end

    # pre-generate random numbers for parallel bootstrap runs
    (opts.trace > 0) && println("creating random number generators for each bootstrap run (using randjump)")
    boot_rngs = gen_boot_rngs(nboot, rng_initial_seed)

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
        boot_weights_fn = (rng, data) -> boot_weights_cluster(rng, crosswalk_df.___idx_cluster___, length(cluster_values))
    else
        @assert isa(boot_weights, Function) "boot_weights must be :simple, :cluster, or a function(rng, data)"
        boot_weights_fn = boot_weights
    end

    all_boot_weights = []
    for i=1:nboot
        (opts.trace > 0) && println("generating bootstrap weights for run ", i)

        # the output is a vector of weights same size as the number of rows from the moment
        boot_weights = boot_weights_fn(boot_rngs[i], data)

        # normalizing weights is important for numerical precision
        boot_weights ./= mean(boot_weights)

        push!(all_boot_weights, boot_weights)
    end

    ### bootstrap moment function 
    # need recenter so that it equal zero at theta_hat
    # see Hall and Horowitz (1996) for details
    m = mom_fn(data, myfit.theta_hat)
    mom_fn_boot = (data, theta) -> mom_fn(data, theta) .- mean(m, dims=1)

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

    # save results to file?
    (opts.path != "") && write(boot_fits, opts)

    # delete all intermediate files with individual iteration results
    if opts.clean_iter 
        print("Deleting individual boot files...")
        rm(opts.path * "__boot__/", force=true, recursive=true)
        println(" Done.")
    end

    # store results
    myfit.vcov = Dict(
        :method => bayesian_bootstrap(),
        :boot_fits => boot_fits,
        :V => cov(boot_fits.all_theta_hat),
        :N => size(m, 1) # TODO : should really move this up to main fit object
    )

    return boot_fits
end

function bboot(
    idx::Int64, 
    data, 
    mom_fn::Function,
    theta0,
    boot_weights; # ? where do we want this?
    W=I,
    opts::GMMOptions)

    (opts.trace > 0) && println("bootstrap run ", idx)

    # path for saving results
    new_opts = copy(opts)
    (new_opts.path != "") && (new_opts.path *= "__boot__/boot_" * string(idx) * "_")

    return fit(data, mom_fn, theta0, W=W, weights=boot_weights, run_parallel=false, opts=new_opts)
end

