

Base.@kwdef mutable struct GMMOptions
    path::String = ""                       # path to save results
    write_iter::Bool = false    # write to file each result (each initial run)
    clean_iter::Bool = false    # delete individual run files at the end of the estimation
    overwrite::Bool = false     # overwrite existing results file and individual run files

    optimizer::Symbol = :optim              # optimizer backend: :optim or :lsqfit (LM)
    optim_algo = LBFGS()                    # Optim.jl algorithm
    optim_opts = nothing                    # additional options. For Optim.jl, this is an Optim.options() object. For LsqFit.jl, this is a NamedTuple
    optim_autodiff::Symbol = :none          # :none or :forward
    theta_lower = nothing                   # nothing or vector of lower bounds
    theta_upper = nothing                   # nothing or vector of upper bounds
    

    theta_names::Union{Vector{String}, Nothing} = nothing  # names of parameters 

    trace::Integer = 0
end

function default_optim_opts()
    return Optim.Options(
        show_trace = false, 
        extended_trace = false,
        iterations=5000)
end


Base.@kwdef mutable struct GMMFit
    theta0::Vector         # initial conditions   (vector of size P or K x P matrix for K sets of initial conditions)
    theta_hat::Vector      # estimated parameters (vector of size P)
    theta_names::Union{Vector{String}, Nothing}

    moms_hat = nothing # value of moments at theta_hat (N x M matrix)
    n_obs = nothing # number of observations (N)
    n_moms = nothing # number of moments (M)

    # estimation parameters
    mode::Symbol = :unassigned # onestep, twostep, etc.
    weights=nothing # Vector of size N or nothing
    W=I             # weight matrix (N x N) or uniform scaling identity (I)
    
    # fit_step1=nothing # for two-step estimation, save results from step 1 # ? need this?

    # optimization results
    obj_value::Number
    converged::Bool
    iterations::Integer
    iteration_limit_reached::Bool
    time_it_took::Float64

    # results from multiple initial conditions (DataFrame)
    all_model_fits = nothing # TODO: switch to PrettyTables.jl and OrderedDict
    idx = nothing # aware of which iteration number this is

    # variance covariance matrix
    vcov = nothing
end


"""
Select best results from multiple initial conditions
"""
function process_model_fits(model_fits::Vector{GMMFit})
    obj_values = [er.obj_value for er = model_fits]
    idx = argmin(obj_values)

    best_model_fit = model_fits[idx]

    best_model_fit.all_model_fits = vcat([table(er) for er = model_fits]...)
    best_model_fit.all_model_fits[!, :is_optimum] = ((1:length(model_fits)) .== idx) .+ 0
    
    return best_model_fit
end

"""
Convert GMMFit object to table
"""
function table(r::GMMFit)

    if !isnothing(r.all_model_fits)
        return r.all_model_fits
    end

    all_model_fits = DataFrame(
        "idx" => [isnothing(r.idx) ? 1 : r.idx],
        "obj_value" => [r.obj_value],
        "converged" => [r.converged],
        "iterations" => [r.iterations],
        "iteration_limit_reached" => [r.iteration_limit_reached],
        "time_it_took" => [r.time_it_took],
    )
    
    # estimated parameters
    nparams = length(r.theta_hat)
    all_model_fits[!, "theta_hat"] = [r.theta_hat]

    # initial conditions
    all_model_fits[!, "theta0"] = [r.theta0]

    return all_model_fits
end

function clean_iter(opts)
    try
        if isdir(opts.path * "__iter__/")
            (opts.trace > 0) && print("Deleting intermediate files from: ", opts.path)
            rm(opts.path * "__iter__/", force=true, recursive=true)
            (opts.trace > 0) && println(" Done.")
        else
            (opts.trace > 0) && println("No intermediate files to delete.")
        end
    catch e
        println(" Error while deleting intermediate files from : ", opts.path, ". Error: ", e)
    end
end

function stats_at_theta_hat(myfit::GMMFit, data, mom_fn::Function)
    
    theta_hat = myfit.theta_hat
    m = mom_fn(data, theta_hat)
    
    myfit.n_obs = size(m, 1)
    myfit.n_moms = size(m, 2)
    
    # TODO: make this optional
    myfit.moms_hat = m
end



###### GMM


"""
"""
function fit(
    data, 
    mom_fn::Function,
    theta0;
    W=I,    
    weights=nothing,
    mode=:onestep,
    run_parallel=true, 
    opts=GMMTools.GMMOptions())

    if mode == :onestep
        return fit_onestep(
                data, 
                mom_fn,
                theta0;
                W=W,    
                weights=weights,
                run_parallel=run_parallel,
                opts=opts)
    else
        @assert mode == :twostep "mode must be :onestep or :twostep"
        return fit_twostep(
                data, 
                mom_fn,
                theta0;
                W=W,    
                weights=weights,
                run_parallel=run_parallel,
                opts=opts)
    end
end

"""
Two-step GMM estimation (one or multiple initial conditions)
"""
function fit_twostep(
    data, 
    mom_fn::Function,
    theta0;
    W=I,    
    weights=nothing,
    run_parallel=true, 
    opts=nothing)

    # avoid modifying the original (change paths)
    opts = deepcopy(opts)

    main_path = opts.path
    (main_path[end] != '/') && (main_path *= "/")

    ### Step 1
    (opts.trace > 0) && println(">>> Starting GMM step 1.")
    opts.path = main_path * "step1/"
    isdir(opts.path) || mkdir(opts.path)

    fit_step1 = fit_onestep(
        data, 
        mom_fn,
        theta0;
        W=W,    
        weights=weights,
        run_parallel=run_parallel,
        opts=opts)

    ### optimal weight matrix
    opts.path = main_path * "step2/"
    isdir(opts.path) || mkdir(opts.path)

    Wstep2_path = opts.path * "Wstep2.csv"
    if isfile(Wstep2_path)
        (opts.trace > 0) && print(">>> Starting GMM step 2. Reading optimal weight matrix from file... ")
        Wstep2 = readdlm(Wstep2_path, ',', Float64)
        (opts.trace > 0) && println("DONE")

    else
        (opts.trace > 0) && print(">>> Starting GMM step 2. Computing optimal weight matrix... ")

        m = mom_fn(data, fit_step1.theta_hat)
        nmomsize = size(m, 1)
        # (opts.trace > 0) && println("number of observations: ", nmomsize)

        Wstep2 = Hermitian(transpose(m) * m / nmomsize)
        Wstep2 = inv(Wstep2)

        # Save Wstep2 to file
        writedlm(Wstep2_path,  Wstep2, ',')
        (opts.trace > 0) && println("DONE and saved to file")
    end

    ### Step 2
    (opts.trace > 0) && println(">>> Starting GMM step 2.")

    fit_step2 = fit_onestep(
        data, 
        mom_fn,
        theta0;
        W=Wstep2, # use optimal weighting matrix from step 1
        weights=weights,
        run_parallel=run_parallel,
        opts=opts)

    # fit_step2.fit_step1 = fit_step1

    # revert
    # opts.path = main_path

    return fit_step2
end


"""
overall: one or multiple initial conditions
"""
function fit_onestep(
    data, 
    mom_fn::Function,
    theta0;
    W=I,    
    weights=nothing,
    run_parallel=true, 
    opts=nothing)


    ### initial conditions
        # number of initial conditions (and always format as matrix, rows=iterations, columns=paramters)
        isa(theta0, Vector) && (theta0 = Matrix(transpose(theta0)))
        nic = size(theta0, 1)
        
        # number of parameters
        # nparam = size(theta0, 2)

    # create folder to store iterations
    if (opts.path != "") && (opts.write_iter)
        (opts.trace > 0) && println("creating path for saving results")
        iterpath = opts.path * "__iter__/"
        isdir(iterpath) || mkdir(iterpath)
    end

    # skip if output file already exists
    if !opts.overwrite && (opts.path != "")
        
        opt_results_from_file = read_fit(opts)
        
        if !isnothing(opt_results_from_file)
            println(" Results file already exists. Reading from file.")

            # delete all intermediate files with individual iteration results
            opts.clean_iter && clean_iter(opts)

            return opt_results_from_file
        end
    end
        
    if (nic == 1) || !run_parallel
        all_model_fits = Vector{GMMFit}(undef, nic)
        for i=1:nic
            all_model_fits[i] = fit_onerun(i, data, mom_fn, theta0[i, :], W=W, weights=weights, opts=opts)
        end

    else
        all_model_fits = @showprogress pmap( i -> fit_onerun(i, data, mom_fn, theta0[i, :], W=W, weights=weights, opts=opts), 1:nic)
    end
    
    best_model_fit = process_model_fits(all_model_fits)
    stats_at_theta_hat(best_model_fit, data, mom_fn)

    # save results to file?
    (opts.path != "") && write(best_model_fit, opts)

    # delete all intermediate files with individual iteration results
    opts.clean_iter && clean_iter(opts)

    return best_model_fit
end

"""
fit function with one initial condition
"""
function fit_onerun(
            idx::Int64, 
            data, 
            mom_fn::Function,
            theta0;
            W=I,    
            weights=nothing, 
            opts::GMMOptions)
            
    (opts.trace > 0) && print("...estimation run ", idx, ". ")

    # skip if output file already exists
    if !opts.overwrite && (opts.path != "")
        
        opt_results_from_file = read_fit(opts, subpath="__iter__/results_" * string(idx))
        
        if !isnothing(opt_results_from_file)
            (opts.trace > 0) && println(" Reading from file.")
            return opt_results_from_file
        end
    end

    if opts.optimizer == :optim
        # Use the general purpose Optim.jl package for optimization (default)

        model_fit = backend_optimjl( 
            data, 
            mom_fn,
            theta0;
            W=W,    
            weights=weights, 
            opts=opts)
        
        model_fit.idx = idx

    elseif opts.optimizer == :lsqfit
        # use the Levenberg Marquardt algorithm from LsqFit.jl for optimization
        # this relies on the fact that the GMM objective function is a sum of squares

        model_fit = backend_lsqfit( 
            data, 
            mom_fn,
            theta0;
            W=W,    
            weights=weights, 
            opts=opts)
        
        model_fit.idx = idx

    else
        error("Optimizer " * string(opts.optimizer) * " not supported. Stopping.")
    end

    # write intermediate results to file
    if opts.write_iter 
        (opts.trace > 0) && println(" Done and done writing to file.")
        write(model_fit, opts, subpath="__iter__/results_" * string(idx)) # this does not contain moms_hat (good, saves space)
    else
        (opts.trace > 0) && println(" Done. ")
    end

    return model_fit
end











