
function backend_optimizer(
    idx,
    data, 
    mom_fn::Function,
    theta0;
    W,    
    weights=nothing, 
    opts::GMMOptions)

    @assert in(opts.optimizer, [:optim, :lsqfit]) "Optimizer " * string(opts.optimizer) * " not supported. Stopping."

    try
        if opts.optimizer == :optim
            # Use the general purpose Optim.jl package for optimization (default)

            return backend_optimjl( 
                data, 
                mom_fn,
                theta0;
                W=W,    
                weights=weights, 
                opts=opts)

        elseif opts.optimizer == :lsqfit
            # use the Levenberg Marquardt algorithm from LsqFit.jl for optimization
            # this relies on the fact that the GMM objective function is a sum of squares

            return backend_lsqfit( 
                data, 
                mom_fn,
                theta0;
                W=W,    
                weights=weights, 
                opts=opts)
        end

    catch e
        # save error to file
        if opts.path != ""
            # write string S to file F
            error_path = opts.path * "__iter__/"
            isdir(error_path) || mkdir(error_path)
            error_path *= "error_" * string(idx) * ".txt"
            
            open(error_path, "w") do io
                Base.write(io, string(e))
            end            
        end

        # do NOT clean iteration files
        opts.clean_iter = false

        # throw?
        if opts.throw_errors 

            throw(e)
        else

            @error "Error in estimation run " * string(idx) * " with theta0=" * string(theta0) *  ". Error: " * string(e)
        end

        return error_fit(e, theta0, W, weights, opts)
    end
end


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
using Optim.jl as backend for optimization
"""
function backend_optimjl( 
            data, 
            mom_fn::Function,
            theta0;
            W=I,    
            weights=nothing, 
            opts::GMMOptions)

    # load the data, W and weights in the moment function 
    gmm_objective_loaded = theta -> gmm_objective(theta, data, mom_fn, W, weights, trace=opts.trace)

    # Optim.jl optimize
    optim_main_args = []
    push!(optim_main_args, gmm_objective_loaded)

    if !isnothing(opts.theta_lower) 
        @assert !isnothing(opts.theta_upper) "if theta_lower is specified, theta_upper must be specified as well"
        push!(optim_main_args, opts.theta_lower)
        push!(optim_main_args, opts.theta_upper)
    end
    push!(optim_main_args, theta0)
    
    # algorithm
    push!(optim_main_args, opts.optim_algo) # defalut = LBFGS()

    # options -- must be a Optim.Options object
    !isnothing(opts.optim_opts) && push!(optim_main_args, opts.optim_opts)

    # autodiff
    if opts.optim_autodiff != :none
        kwargs = (autodiff = :forward, )
    else
        kwargs = ()
    end

    time_it_took = @elapsed raw_opt_results = Optim.optimize(optim_main_args...; kwargs...)

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
        time_it_took = time_it_took
    )

    return model_fit
end


############### LSQFIT (write the objective as a sum of squares)

function gmm_objective_half(theta::Vector, data, mom_fn::Function, Whalf, weights)

    m = mom_fn(data, theta)
    @assert isa(m, Matrix) "m(data, theta) must return a Matrix (rows = observations, columns = moments)"
    
    # average of the moments over all observations (with/without weights)
    if isnothing(weights)
        mmean = mean(m, dims=1)
    else
        # weights is a vector, so result here is (1 x n_moms)
        mmean = (weights' * m) ./ sum(weights)
    end

    # objective
    return vec(mmean * Whalf)
end

"""
using LsqFit.jl as backend for optimization, especially the Levenberg-Marquardt algorithm
"""
function backend_lsqfit(
            data, 
            mom_fn::Function,
            theta0;
            W=I,    
            weights=nothing, 
            opts::GMMOptions)


    # Cholesky decomposition of W = Whalf * Whalf' 
    if !isa(W, UniformScaling)
        Whalf = Matrix(cholesky(Hermitian(W)).L)
        if norm(Whalf * transpose(Whalf) - W) > 1e-8
            @warn "Cholesky decomposition approximate: abs(Whalf * Whalf' - W) > 1e-8"
        end
    else
        Whalf = W
    end

    # Objective function: multiply avg moments by (Cholesky) half matrice and take means
    # (1 x n_moms) x (n_moms x n_moms) = (1 x n_moms)
    # gmm_objective_loaded = (x, theta) -> vec(mean(mom_fn(data, theta), dims=1) * Whalf)
    gmm_objective_loaded = (x, theta) -> gmm_objective_half(theta, data, mom_fn, Whalf, weights)

    # Build options programatically
    m = mom_fn(data, theta0)
    n_moms = size(m, 2)

    lsqfit_main_args = []
    push!(lsqfit_main_args, gmm_objective_loaded) # function
    push!(lsqfit_main_args, zeros(n_moms)) # x
    push!(lsqfit_main_args, zeros(n_moms)) # y
    push!(lsqfit_main_args, theta0)

    mynames = []
    myvalues = []
    if !isnothing(opts.theta_lower)
        @assert !isnothing(opts.theta_upper) "if theta_lower is specified, theta_upper must be specified as well"

        push!(mynames, :lower)
        push!(myvalues, opts.theta_lower)

        push!(mynames, :upper)
        push!(myvalues, opts.theta_upper)
    end

    if opts.optim_autodiff != :none
        push!(mynames, :autodiff)
        push!(myvalues, :forwarddiff)
    end

    # always store the trace (to know how many iterations were done)
    push!(mynames, :store_trace)
    push!(myvalues, true)

    lsqfit_kwargs = NamedTuple{Tuple(mynames)}(myvalues)

    if !isnothing(opts.optim_opts)
        lsqfit_kwargs = merge(lsqfit_kwargs, opts.optim_opts)
    end
    # push!(mynames, :show_trace)
    # push!(myvalues, true)

    time_it_took = @elapsed raw_opt_results = curve_fit(lsqfit_main_args...; lsqfit_kwargs...)

    model_fit = GMMFit(
        converged = raw_opt_results.converged,
        theta_hat = raw_opt_results.param,
        theta_names = opts.theta_names,
        weights=weights,
        W=W,
        obj_value = sum(raw_opt_results.resid .* raw_opt_results.resid),
        iterations = length(raw_opt_results.trace),  
        iteration_limit_reached = length(raw_opt_results.trace) >= 1000,
        theta0 = theta0,
        time_it_took = time_it_took
    )
    return model_fit
end
