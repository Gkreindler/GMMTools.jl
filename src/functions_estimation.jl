
Base.@kwdef mutable struct GMMFit
    theta0::Vector
    theta_hat::Vector
    theta_names::Union{Vector{String}, Nothing}

    # estimation parameters
    mode::Symbol = :unassigned # onestep, twostep, etc.
    weights=nothing
    W=I
    N = -1 # number of observations
    
    fit_step1=nothing

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

function Base.show(io::IO, r::GMMFit)
    println("GMMResult object with fields: thata0, W, weights, N, all_model_fits, vcov, fit_step1, etc.")
    println("  theta_names:", r.theta_names)
    println("  thata_hat:  ", r.theta_hat)
    println("Optimization results:")
    println("  converged:  ", r.converged)
    println("  obj_value:  ", r.obj_value)
    println("  iterations:  ", r.iterations)
    println("  iteration_limit_reached:  ", r.iteration_limit_reached)
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


Base.@kwdef mutable struct GMMOptions
    path::String = ""
    theta_names::Union{Vector{String}, Nothing} = nothing
    optim_autodiff::Symbol = :none
    optim_algo = LBFGS()
    optim_opts = nothing
    write_iter::Bool = false    # write to file each result (each initial run)
    clean_iter::Bool = false    # 
    overwrite::Bool = false
    trace::Integer = 0
end

function default_optim_opts()
    return Optim.Options(
        show_trace = false, 
        extended_trace = false,
        iterations=5000)
end

function default_gmm_opts(;
    path = "",
    theta_names = nothing,
    optim_opts = default_optim_opts(),
    optim_autodiff = :none,
    optim_algo = LBFGS(),
    write_iter = false,    # write to file each result (each initial run)
    clean_iter = false,    # 
    overwrite = false,
    trace = 0)

    return GMMOptions(
        path=path,
        theta_names=theta_names,
        optim_autodiff=optim_autodiff,
        optim_algo=optim_algo,
        optim_opts=optim_opts,
        write_iter=write_iter,
        clean_iter=clean_iter,
        overwrite=overwrite,
        trace=trace)
end

# extent the "copy" method to the GMMOptions type
Base.copy(x::GMMOptions) = GMMOptions([getfield(x, k) for k ∈ fieldnames(GMMOptions)]...)

function write(est_result::GMMFit, opts::GMMOptions, filename; subpath="")
    
    if opts.path == ""
        return
    end

    if subpath == ""
        full_path = opts.path
    else
        (subpath[end] != '/') && (subpath *= "/")
        full_path = opts.path * subpath
        isdir(full_path) || mkdir(full_path)
    end

    CSV.write(full_path * filename, table(est_result))
end


function parse_vector(s::AbstractString)
    return parse.(Float64, split(s[2:(end-1)],","))
end

function load_from_file(opts::GMMOptions; filepath="")

    if filepath == ""
        full_path = opts.path * "results.csv"
    else
        full_path = opts.path * filepath
    end

    if isfile(full_path)
        df = CSV.read(full_path, DataFrame)

        if nrow(df) == 1
            return GMMFit(
                theta0=parse_vector(df[1, :theta0]),
                theta_hat=parse_vector(df[1, :theta_hat]),
                theta_names=opts.theta_names,
                converged=df[1, :converged],
                obj_value=df[1, :obj_value],
                iterations=df[1, :iterations],
                iteration_limit_reached=df[1, :iteration_limit_reached],
                time_it_took=df[1, :time_it_took],
                idx=df[1, :idx])
        
        else
            
                mysample = df.is_optimum .== 1
                df_optimum = df[mysample, :]

            return GMMFit(
                theta0=parse_vector(df_optimum[1, :theta0]),
                theta_hat=parse_vector(df_optimum[1, :theta_hat]),
                theta_names=opts.theta_names,
                converged=df_optimum[1, :converged],
                obj_value=df_optimum[1, :obj_value],
                iterations=df_optimum[1, :iterations],
                iteration_limit_reached=df_optimum[1, :iteration_limit_reached],
                time_it_took=df_optimum[1, :time_it_took],
                idx=df_optimum[1, :idx],
                all_model_fits=df)

            # @error "load large DF not yet supported"
        end
    else
        return nothing
    end
end

function clean_iter(opts)
    try
        (opts.trace > 0) && print("Deleting intermediate files from: ", opts.path)
        rm(opts.path * "__iter__/", force=true, recursive=true)
        (opts.trace > 0) && println(" Done.")
    catch e
        println(" Error while deleting intermediate files from : ", opts.path, ". Error: ", e)
    end
end

function add_nobs(myfit::GMMFit, data, mom_fn::Function, theta0)
    try
        myfit.N = size(data, 1)
    catch
        mytheta = theta0[1, :]
        myfit.N = size(mom_fn(data, mytheta), 1)
    end
end



###### GMM

function gmm_objective(theta::Vector, data, mom_fn::Function, W, weights; trace=0)

    t1 = @elapsed m = mom_fn(data, theta)

    @assert isa(m, Matrix) "m(data, theta) must return a Matrix (rows = observations, columns = moments)"

    (trace > 1) && println("Evaluation took ", t1)
    
    # average of the moments over all observations (with/without weights)
    if isnothing(weights)
        mmean = mean(m, dims=1)
    else
        mmean = (weights' * m) ./ sum(weights)
    end

    # objective
    return (mmean * W * Transpose(mmean))[1]
end

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
    opts=default_gmm_opts())

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
    opts=default_gmm_opts())

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
    m = mom_fn(data, fit_step1.theta_hat)
    nmomsize = size(m, 1)
    # (opts.trace > 0) && println("number of observations: ", nmomsize)

    Wstep2 = Hermitian(transpose(m) * m / nmomsize)
    Wstep2 = inv(Wstep2)

    # Save Wstep2 to file
    writedlm( opts.path * "Wstep2.csv",  Wstep2, ',')

    ### Step 2
    (opts.trace > 0) && println(">>> Starting GMM step 2.")
    opts.path = main_path * "step2/"
    isdir(opts.path) || mkdir(opts.path)

    fit_step2 = fit_onestep(
        data, 
        mom_fn,
        theta0;
        W=Wstep2,    
        weights=weights,
        run_parallel=run_parallel,
        opts=opts)

    fit_step2.fit_step1 = fit_step1

    # revert
    opts.path = main_path

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
    opts=default_gmm_opts())

    # number of initial conditions (and always format as matrix, rows=iterations, columns=paramters)
    nic = size(theta0, 1)
    # number of parameters
    nparam = size(theta0, 2)

    # skip if output file already exists
    if !opts.overwrite && (opts.path != "")
        
        opt_results_from_file = load_from_file(opts)
        
        if !isnothing(opt_results_from_file)
            println(" Results file already exists. Reading from file.")
            add_nobs(opt_results_from_file, data, mom_fn, theta0)

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
        all_model_fits = pmap( i -> fit_onerun(i, data, mom_fn, theta0[i, :], W=W, weights=weights, opts=opts), 1:nic)
    end
    
    best_model_fit = process_model_fits(all_model_fits)
    add_nobs(best_model_fit, data, mom_fn, theta0)

    # save results to file?
    (opts.path != "") && write(best_model_fit, opts, "results.csv")

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

    # default optimizer options (if missing)
    isnothing(opts.optim_opts) && (opts.optim_opts = default_optim_opts())

    (opts.trace > 0) && print("...estimation run ", idx, ". ")

    # skip if output file already exists
    if !opts.overwrite && (opts.path != "")
        
        opt_results_from_file = load_from_file(opts, filepath="__iter__/results_" * string(idx) * ".csv")
        
        if !isnothing(opt_results_from_file)
            (opts.trace > 0) && println(" Reading from file.")
            return opt_results_from_file
        end
    end

    # load the data
    gmm_objective_loaded = theta -> gmm_objective(theta, data, mom_fn, W, weights, trace=opts.trace)

    # optimize
    if opts.optim_autodiff == :forward
        (opts.trace > 0) && print("using AD")
        time_it_took = @elapsed raw_opt_results = Optim.optimize(gmm_objective_loaded, 
                                                                theta0, 
                                                                opts.optim_algo, # defalut = LBFGS()
                                                                opts.optim_opts, 
                                                                autodiff=:forward)

        # results = optimize(f, g!, lower, upper, initial_x, Fminbox(GradientDescent()), Optim.Options(outer_iterations = 2))

    else
        time_it_took = @elapsed raw_opt_results = Optim.optimize(gmm_objective_loaded, 
                                                                theta0, 
                                                                opts.optim_algo, # defalut = LBFGS()
                                                                opts.optim_opts)
    end
    #= 
    summary(res)
    minimizer(res)
    minimum(res)
    iterations(res)
    iteration_limit_reached(res)
    trace(res)
    x_trace(res)
    f_trace(res)
    f_calls(res)
    converged(res)
    =#
    
    model_fit = GMMFit(
        converged = Optim.converged(raw_opt_results),
        theta_hat = Optim.minimizer(raw_opt_results),
        theta_names = opts.theta_names,
        weights=weights,
        W=W,
        obj_value = Optim.minimum(raw_opt_results),
        iterations = Optim.iterations(raw_opt_results),
        iteration_limit_reached = Optim.iteration_limit_reached(raw_opt_results),
        theta0 = theta0,
        time_it_took = time_it_took,
        idx=idx
    )

    # write intermediate results to file
    if opts.write_iter 
        (opts.trace > 0) && println(" Done and done writing to file.")
        write(model_fit, opts, "results_" * string(idx) * ".csv", subpath="__iter__")
    else
        (opts.trace > 0) && println(" Done. ")
    end

    return model_fit
end











