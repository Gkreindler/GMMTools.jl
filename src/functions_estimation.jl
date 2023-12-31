

Base.@kwdef mutable struct GMMOptions

    # files
    path::String = ""                       # path to save results
    write_iter::Bool = false    # write to file each result (each initial run)
    clean_iter::Bool = false    # delete individual run files at the end of the estimation
    overwrite::Bool = true      # overwrite existing results file and individual run files
    throw_errors::Bool = true   # throw optimization errors (if false, save them to file but continue with the other runs)

    # optimizer
    optimizer::Symbol = :optim              # optimizer backend: :optim or :lsqfit (LM)
    optim_algo = LBFGS()                    # Optim.jl algorithm
    optim_opts = nothing                    # additional options. For Optim.jl, this is an Optim.options() object. For LsqFit.jl, this is a NamedTuple
    optim_autodiff::Symbol = :none          # :none or :forward
    theta_lower = nothing                   # nothing or vector of lower bounds
    theta_upper = nothing                   # nothing or vector of upper bounds
    
    # parameter
    theta_factors::Union{Vector{Float64}, Nothing} = nothing # options are nothing or a vector of length P with factors for each parameter. Parameter theta[i] will be replaced by theta[i] * theta_factors[i] before optimization
    theta_names::Union{Vector{String}, Nothing} = nothing  # names of parameters 

    # display
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
    theta_factors::Union{Vector{Float64}, Nothing} = nothing # nothing or a vector of length P with factors for each parameter. Parameter theta[i] was replaced by theta[i] * theta_factors[i] before optimization

    moms_hat = nothing # value of moments at theta_hat (N x M matrix)
    n_obs = nothing # number of observations (N)
    n_moms = nothing # number of moments (M)

    # estimation parameters
    mode::Symbol = :unassigned # onestep, twostep1, twostep2,  etc.
    weights=nothing # Vector of size N or nothing
    W=I             # weight matrix (N x N) or uniform scaling identity (I)
    
    # fit_step1=nothing # for two-step estimation, save results from step 1 # ? need this?

    # optimization results
    obj_value::Number
    errored::Bool = false
    error_message::String = ""
    converged::Bool
    iterations::Union{Integer, Missing}
    iteration_limit_reached::Union{Bool, Missing}
    time_it_took::Union{Float64, Missing}

    # results from multiple initial conditions (DataFrame)
    fits_df = nothing
    idx = nothing # aware of which iteration number this is

    # variance covariance matrix
    vcov = nothing
end

"""
The model fit object when the optimization gives an error
"""
function error_fit(e, theta0, W, weights, mode, opts)
    return GMMFit(
        mode=mode,
        errored = true,
        error_message = string(e),
        converged = false,
        theta_hat = missing .* theta0,
        theta_names = opts.theta_names,
        weights=weights,
        W=W,
        obj_value = Inf, # pick Inf so we never pick this as the best iteration (unless all iterations errored)
        iterations = missing,
        iteration_limit_reached = missing,
        theta0 = theta0,
        time_it_took = missing
    )
end

"""
Select best results from multiple initial conditions
    - errored runs have obj_value = Inf. Hence, if all runs errored, we (arbitrarily) pick idx=1 as the best run, and it will have errored = true
"""
function process_model_fits(model_fits::Vector{GMMFit})

    # # vector of errors
    # errors = [er.errored for er = model_fits]
    
    # # smallest objective value among non-error runs
    # obj_values = [model_fits[i].obj_value for i=1:length(model_fits) if !errors[i]]
    # if isempty(obj_values)
    #     idx = 1
    # else
    #     idx = argmin(obj_values)
    # end    

    obj_values = [model_fits[i].obj_value for i=1:length(model_fits)]    
    idx = argmin(obj_values)

    best_model_fit = model_fits[idx]

    best_model_fit.fits_df = vcat([table_fit(er) for er = model_fits]...)
    best_model_fit.fits_df[!, :is_optimum] = ((1:length(model_fits)) .== idx) .+ 0
    
    return best_model_fit
end

"""
Convert GMMFit object to table
"""
function table_fit(r::GMMFit)

    if !isnothing(r.fits_df)
        return r.fits_df
    end

    fits_df = DataFrame(
        "idx" => [isnothing(r.idx) ? 1 : r.idx],
        "obj_value" => [r.obj_value],
        "converged" => [r.converged],
        "errored" => [r.errored],
        "iterations" => [r.iterations],
        "iteration_limit_reached" => [r.iteration_limit_reached],
        "time_it_took" => [r.time_it_took],
    )
    
    # estimated parameters
    nparams = length(r.theta_hat)
    fits_df[!, "theta_hat"] = [r.theta_hat]

    # initial conditions
    fits_df[!, "theta0"] = [r.theta0]

    return fits_df
end



function stats_at_theta_hat(myfit::GMMFit, data, mom_fn::Function)
    
    theta_hat = myfit.theta_hat

    if !myfit.errored
        m = mom_fn(data, theta_hat)
        myfit.n_obs = size(m, 1)
        myfit.n_moms = size(m, 2)
        
        # TODO: make this optional
        myfit.moms_hat = m
    end
end


###### GMM
"""
gateway function to estimate GMM model
"""
function fit(
    data,               # any object that can be passed to mom_fn as the first argument
    mom_fn::Function,   # mom_fn(data, theta) returns a matrix of moments (N x M)
    theta0;             # initial conditions (vector of size P or K x P matrix for K sets of initial conditions)
    W=I,                # weight matrix (N x N) or uniform scaling identity (I)
    weights=nothing,    # Vector of size N or nothing
    mode=:onestep,      # :onestep or :twostep
    run_parallel=false, # run in parallel (pmap, embarasingly parallel) or not
    opts=GMMTools.GMMOptions() # other options
)

    # checks # TODO: add more
    @assert isa(mom_fn, Function) "mom_fn must be a function"
    @assert isa(theta0, Vector) || isa(theta0, Matrix) "theta0 must be a Vector (P) or a Matrix (K x P)"
    @assert isa(W, Matrix) || isa(W, UniformScaling) "W must be a Matrix or UniformScaling (e.g. I)"
    @assert isa(weights, Vector) || isnothing(weights) "weights must be a Vector or nothing"
    @assert mode == :onestep || mode == :twostep "mode must be :onestep or :twostep"    

    # one-step or two-step GMM
    if mode == :onestep
        return fit_onestep(
                data, 
                mom_fn,
                theta0;
                W=W,    
                weights=weights,
                run_parallel=run_parallel,
                mode=:onestep,
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
        myfit.mode = :twostep
        return myfit
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
        mode=:twostep1,
        opts=opts)

    # path for step 2
    opts.path = main_path * "step2/"
    isdir(opts.path) || mkdir(opts.path)

    # if step1 errored
    if fit_step1.errored
        fit_step2 = deepcopy(fit_step1)
        fit_step2.mode = :twostep2
        # save results to file?
        (opts.path != "") && write(fit_step2, opts.path)

        return fit_step2
    end

    ### optimal weight matrix
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
        mode=:twostep2,
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
    mode=nothing,
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
        fits_df = Vector{GMMFit}(undef, nic)
        for i=1:nic
            fits_df[i] = fit_onerun(i, data, mom_fn, theta0[i, :], W=W, weights=weights, mode=mode, opts=opts)
        end

    else
        fits_df = @showprogress pmap( i -> fit_onerun(i, data, mom_fn, theta0[i, :], W=W, weights=weights, mode=mode, opts=opts), 1:nic)
    end
    
    best_model_fit = process_model_fits(fits_df)
    stats_at_theta_hat(best_model_fit, data, mom_fn)

    # save results to file?
    (opts.path != "") && write(best_model_fit, opts.path)

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
            mode,
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

    # try/catch block and select optimizer
    model_fit = backend_optimizer(
        idx,
        data, 
        mom_fn,
        theta0;
        W=W,    
        weights=weights, 
        mode=mode,
        opts=opts)
    
    model_fit.idx = idx

    # write intermediate results to file
    if opts.write_iter 
        (opts.trace > 0) && println(" Done and done writing to file.")
        write(model_fit, opts.path, subpath="__iter__/results_" * string(idx)) # this does not contain moms_hat (good, saves space)
    else
        (opts.trace > 0) && println(" Done. ")
    end

    return model_fit
end











