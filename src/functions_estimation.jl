#= 
TODOS:
1. add try ... catch blocks
1. add optimizer options / variants. Where to save the optimizer-specific options? kwargs.
1. save bootstrap weights?
1. add parameter names in the problem object
1. switch to PrettyTables.jl (cf Peter's suggestion)
=#


Base.@kwdef mutable struct GMMProblem
    data
    cache = nothing
    W=I                 # weight Matrix
    theta0::Array
    weights             # weights for each observation (e.g. used in Bayesian bootstrap)
    theta_names::Union{Vector{String}, Nothing} = nothing
end

function create_GMMProblem(;
    data, 
    cache=nothing, 
    W=I, 
    theta0, 
    weights=nothing,
    theta_names::Union{Vector{String}, Nothing} = nothing)

    if isa(theta0, Vector)
        theta0 = Matrix(Transpose(theta0))
    end

    if isnothing(theta_names)
        nparams = size(theta0, 2)
        theta_names = ["θ_" * string(i) for i=1:nparams]
    end

    return GMMProblem(data, cache, W, theta0, weights, theta_names) 
end

function Base.show(io::IO, prob::GMMProblem)
    println("GMM Problem, fields:")
    println("- data")
    !isnothing(prob.cache) && println("- cache")
    println("- W = weighting matrix for GMM")
    !isnothing(prob.theta_names) && println("- theta_names ", prob.theta_names)
    println("- theta0 = initial conditions ", prob.theta0)
    println("- weights = observation weights ")
end

Base.@kwdef mutable struct GMMResult
    theta0::Vector
    theta_hat::Vector
    theta_names::Vector{String}
    obj_value::Number
    converged::Bool
    iterations::Integer
    iteration_limit_reached::Bool
    time_it_took::Float64
    all_results = nothing # TODO: switch to PrettyTables.jl and OrderedDict
    idx = nothing # aware of which iteration number this is

    N = -1 # number of observations

    vcov = nothing
end

function table(r::GMMResult)

    if !isnothing(r.all_results)
        return r.all_results
    end

    all_results = DataFrame(
        "idx" => [isnothing(r.idx) ? 1 : r.idx],
        "obj_value" => [r.obj_value],
        "converged" => [r.converged],
        "iterations" => [r.iterations],
        "iteration_limit_reached" => [r.iteration_limit_reached],
        "time_it_took" => [r.time_it_took],
    )
    
    # estimated parameters
    nparams = length(r.theta_hat)
    all_results[!, "theta_hat"] = [r.theta_hat]

    # initial conditions
    all_results[!, "theta0"] = [r.theta0]

    return all_results
end

function Base.show(io::IO, r::GMMResult)
    println("θ hat    = ", r.theta_hat)
    println("Converged? ", r.converged)
    println("Obj value  ", r.obj_value)
    display(r.all_results)
end




### Initial conditions

"""
Creates a random matrix of initial conditions, taking bounds into account
"""
function random_theta0(theta0::Vector, nic::Int; theta_lower=nothing, theta_upper=nothing)

    n_params = length(theta0)
    theta0_mat = zeros(nic, n_params)
    isnothing(theta_lower) && (theta_lower = fill(-Inf, n_params))
    isnothing(theta_upper) && (theta_upper = fill( Inf, n_params))

    for i=1:nic
        for j=1:n_params

            θL = theta_lower[j]
            θH = theta_upper[j]

            if (θL == -Inf) & (θH == Inf)
                theta0_mat[i,j] = (-1.5 + 3.0 * rand()) * theta0[j]
            elseif (θL > -Inf) & (θH == Inf)
                theta0_mat[i,j] = θL + (theta0[j] - θL) * (0.5 + rand())
            elseif (θL == -Inf) & (θH < Inf)
                theta0_mat[i,j] = θH - (θH - theta0[j]) * (0.5 + rand())
            else
                @assert (θL > -Inf) & (θH < Inf)
                theta0_mat[i,j] = θL + (θH - θL) * rand()
            end
        end
    end

    return theta0_mat
end


function process_results(est_results::Vector{GMMResult})
    obj_values = [er.obj_value for er = est_results]
    idx = argmin(obj_values)

    best_result = est_results[idx]

    best_result.all_results = vcat([table(er) for er = est_results]...)
    best_result.all_results[!, :is_optimum] = ((1:length(est_results)) .== idx) .+ 0
    
    # sort!(best_result.all_results, :obj_value)

    return best_result
end


Base.@kwdef mutable struct GMMOptions
    path::String = ""
    autodiff::Symbol = :none
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
    optim_opts=default_optim_opts(),
    autodiff=:none,
    write_iter = false,    # write to file each result (each initial run)
    clean_iter = false,    # 
    overwrite = false,
    trace = 0)

    return GMMOptions(
        path=path,
        autodiff=autodiff,
        optim_opts=optim_opts,
        write_iter=write_iter,
        clean_iter=clean_iter,
        overwrite=overwrite,
        trace=trace)
end

# extent the "copy" method to the GMMOptions type
Base.copy(x::GMMOptions) = GMMOptions([getfield(x, k) for k ∈ fieldnames(GMMOptions)]...)

function write(est_result::GMMResult, opts::GMMOptions; subpath="", filename="")
    if opts.path == ""
        return
    end

    if subpath == ""
        full_path = opts.path
    else
        (subpath[end] != "/") && (subpath *= "/")
        full_path = opts.path * subpath
        isdir(full_path) || mkdir(full_path)
    end

    (filename == "") ? filename = "results.csv" : nothing

    file_path = full_path * filename

    CSV.write(file_path, table(est_result))
end


function parse_vector(s::String)
    return parse.(Float64, split(s[2:(end-1)],","))
end

function load(prob::GMMProblem, opts::GMMOptions, filepath::String)

    full_path = opts.path * filepath
    if isfile(full_path)
        df = CSV.read(full_path, DataFrame)

        if nrow(df) == 1
            return GMMResult(
                theta0=parse_vector(df[1, :theta0]),
                theta_hat=parse_vector(df[1, :theta_hat]),
                theta_names=prob.theta_names,
                converged=df[1, :converged],
                obj_value=df[1, :obj_value],
                iterations=df[1, :iterations],
                iteration_limit_reached=df[1, :iteration_limit_reached],
                time_it_took=df[1, :time_it_took],
                idx=df[1, :idx])
        
        else
            
                mysample = df.is_optimum .== 1
                df_optimum = df[mysample, :]

            return GMMResult(
                theta0=parse_vector(df_optimum[1, :theta0]),
                theta_hat=parse_vector(df_optimum[1, :theta_hat]),
                theta_names=prob.theta_names,
                converged=df_optimum[1, :converged],
                obj_value=df_optimum[1, :obj_value],
                iterations=df_optimum[1, :iterations],
                iteration_limit_reached=df_optimum[1, :iteration_limit_reached],
                time_it_took=df_optimum[1, :time_it_took],
                idx=df_optimum[1, :idx],
                all_results=df)

            # @error "load large DF not yet supported"
        end
    else
        return nothing
    end
end

function add_nobs(myfit::GMMResult, problem::GMMProblem, mom_fn::Function)
    try
        myfit.N = size(problem.data, 1)
    catch
        mytheta = problem.theta0[1, :]
        myfit.N = size(mom_fn(problem, mytheta), 1)
    end
end



###### GMM

function gmm_objective(theta::Vector, problem::GMMProblem, mom_fn::Function; trace=0)

    t1 = @elapsed m = mom_fn(problem, theta)

    @assert isa(m, Matrix) "m must be a Matrix (rows = observations, columns = moments)"

    (trace > 1) && println("Evaluation took ", t1)
    
    # average of the moments over all observations (with/without weights)
    if isnothing(problem.weights)
        mmean = mean(m, dims=1)
    else
        mmean = (problem.weights' * m) ./ sum(problem.weights)
    end

    # objective
    return (mmean * problem.W * Transpose(mmean))[1]
end

"""
"""
function fit(
    problem::GMMProblem, 
    mom_fn::Function;
    mode=:onestep,
    run_parallel=true, 
    opts=default_gmm_opts())

    if mode == :onestep
        return fit_onestep(
                problem,
                mom_fn,
                run_parallel=run_parallel,
                opts=opts)
    else
        @assert mode == :twostep "mode must be :onestep or :twostep"
        return fit_twostep(
                problem,
                mom_fn,
                run_parallel=run_parallel,
                opts=opts)
    end
end

"""
Two-step GMM estimation (one or multiple initial conditions)
"""
function fit_twostep(
    problem::GMMProblem, 
    mom_fn::Function;
    run_parallel=true, 
    opts=default_gmm_opts())

    # Step 1
    fit_step1 = fit_onestep(
        problem,
        mom_fn,
        run_parallel=run_parallel,
        opts=opts)

    # optimal weight matrix
    m = mom_fn(problem, fit_step1.theta_hat)
    nmomsize = size(m, 1)
    println("number of observations: ", nmomsize)

    Wstep2 = Hermitian(transpose(m) * m / nmomsize)
    Wstep2 = inv(Wstep2)
    problem.W = Wstep2

    display(Wstep2)

    # TODO: save Wstep2 to file

    fit_step2 = fit_onestep(
        problem,
        mom_fn,
        run_parallel=run_parallel,
        opts=opts)

    return fit_step2
end


"""
overall: one or multiple initial conditions
"""
function fit_onestep(
    problem::GMMProblem, 
    mom_fn::Function;
    run_parallel=true, 
    opts=default_gmm_opts())

    # skip if output file already exists
    if !opts.overwrite && (opts.path != "")
        
        opt_results_from_file = load(problem, opts, "results.csv")
        
        if !isnothing(opt_results_from_file)
            println(" Results already exist. Reading from file.")
            add_nobs(opt_results_from_file, problem, mom_fn)
            return opt_results_from_file
        end
    end

    # number of initial conditions (and always format as matrix, rows=iterations, columns=paramters)
    nic = size(problem.theta0, 1)
        
    if (nic == 1) || !run_parallel
        several_est_results = Vector{GMMResult}(undef, nic)
        for i=1:nic
            several_est_results[i] = fit_onerun(i, problem, mom_fn, opts)
        end
    else
        several_est_results = pmap( i -> fit_onerun(i, problem, mom_fn, opts), 1:nic)
    end
    
    best_result = process_results(several_est_results)
    add_nobs(best_result, problem, mom_fn)

    # save results to file?
    (opts.path != "") && write(best_result, opts, filename="results.csv")

    # delete all intermediate files with individual iteration results
    if opts.clean_iter 
        print("Deleting intermediate files...")
        rm(opts.path * "__iter__/", force=true, recursive=true)
        println(" Done.")
    end

    return best_result
end
"""
fit function with one initial condition
"""
function fit_onerun(idx::Int64, problem::GMMProblem, mom_fn::Function, opts::GMMOptions)

    # default optimizer options (if missing)
    isnothing(opts.optim_opts) && (opts.optim_opts = default_optim_opts())

    (opts.trace > 0) && print("...estimation run ", idx, ". ")

    # skip if output file already exists
    if !opts.overwrite && (opts.path != "")
        
        opt_results_from_file = load(problem, opts, "__iter__/results_" * string(idx) * ".csv")
        
        if !isnothing(opt_results_from_file)
            (opts.trace > 0) && println(" Reading from file.")
            return opt_results_from_file
        end
    end

    # single vector of initial conditions
    theta0 = problem.theta0[idx, :]

    # load the data
    gmm_objective_loaded = theta -> gmm_objective(theta, problem, mom_fn, trace=opts.trace)

    # optimize
    if opts.autodiff == :forward
        println("using AD")
        time_it_took = @elapsed raw_opt_results = Optim.optimize(gmm_objective_loaded, theta0, LBFGS(), opts.optim_opts, autodiff=:forward)
    else
        time_it_took = @elapsed raw_opt_results = Optim.optimize(gmm_objective_loaded, theta0, opts.optim_opts)
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
    
    opt_results = GMMResult(
        converged = Optim.converged(raw_opt_results),
        theta_hat = Optim.minimizer(raw_opt_results),
        theta_names = problem.theta_names,
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
        write(opt_results, opts, subpath="__iter__", filename="results_" * string(idx) * ".csv")
    else
        (opts.trace > 0) && println(" Done. ")
    end

    return opt_results
end











