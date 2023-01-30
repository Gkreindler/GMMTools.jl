

"""
Creates a random matrix of initial conditions, taking boudns into account
"""
function random_initial_conditions(theta0, theta_lower, theta_upper, n_init)

    n_params = length(theta0)

    theta0_mat = zeros(n_init, n_params)

    for i=1:n_init
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

"""
    theta = typically the first stage estimate
    momfn = moment function loaded with data and other parameters
"""
function vcov_gmm_iid(theta, momfn)
    # compute matrix of moments
    mom_matrix = momfn(theta)

    # compute variance covariance matrix under iid assumption
    # ensure it's Hermitian
    n_observations = size(mom_matrix, 1)
    vcov_matrix = Hermitian(transpose(mom_matrix) * mom_matrix / n_observations)

    return vcov_matrix
end

# TODO: describe what this wrapper does
"""
"""
function gmm_obj(;theta, Whalf, momfn, show_theta=false)

	# print parameter vector at current step
	show_theta && println(">>> theta ", theta, " ")

	# compute moments
	mymoms = momfn(theta)

	# multiply by (Cholesky) half matrice and take means
	mean_mymoms = vec(mean(mymoms, dims=1) * Whalf)

	# write the value of the objective function at the end of the line
	show_theta && println(">>> obj value:", transpose(mean_mymoms) * mean_mymoms)

	return mean_mymoms
end


function curve_fit_wrapper(
				idx,
				myobjfunction,
				Whalf,
				n_moms,
				theta_initial_vec,
				theta_lower,
				theta_upper;
				write_results_to_file=0,
				individual_run_results_path="",
                overwrite_existing::String="all",
				maxIter=1000,
				maxTime=Inf,
                throw_errors::Bool=false,
                show_trace::Bool=false,
                show_progress::Bool=true
			)

    show_progress && println("starting iteration ", idx)

    outputfile = individual_run_results_path * "results_df_run_" * string(idx) * ".csv"

    if overwrite_existing != "all" && isfile(outputfile)
        results_df = CSV.read(outputfile, DataFrame)

        has_error = results_df[1, "opt_error"]
        hit_limit = ~results_df[1, "opt_converged"]

        if overwrite_existing == "none"
            show_progress && println("... run exists -- skipping, error=", has_error, " limit=", hit_limit)    
            return results_df
        end

        if ~has_error && ~hit_limit
            show_progress && println("... successful run -- skipping, error=", has_error, " limit=", hit_limit)
            return results_df
        end

        if has_error && (overwrite_existing == "limit")
            show_progress && println("... errored run -- skipping, error=", has_error, " limit=", hit_limit)
            return results_df
        end

        if hit_limit && (overwrite_existing == "error")
            show_progress && println("... run that hit limit -- skipping, error=", has_error, " limit=", hit_limit)
            return results_df
        end
    end

    try
        # 	moments are already differenced out so we target zero:
        ymoms = zeros(n_moms)

        # call curve_fit from LsqFit.jl
        timeittook = @elapsed result = curve_fit(
                    myobjfunction, # objective function, takes theta and Whalf as arguments
                    Whalf, # pass cholesky half as "data" to myobjfunction
                    ymoms, # zeros
                    theta_initial_vec,
                    lower=theta_lower,
                    upper=theta_upper,
                    maxIter=maxIter,
                    maxTime=maxTime, # this is added relative to official LsqFit.jl . Use https://github.com/Gkreindler/LsqFit.jl
                    show_trace=show_trace
                )

        # TODO: remove this?
        if show_progress
            println(">>> iteration ", idx, " took: ", timeittook, " seconds. Converged: ", result.converged)
            println(">>> optimum ", result.param)
            println(">>> obj val ", norm(result.resid))
        end

        # results dictionary
        results_df = Dict{String, Any}(
            "obj_vals" => norm(result.resid),
            "opt_converged" => result.converged,
            "opt_runtime" => timeittook,
            "opt_error" => false
        )
        for i=1:length(result.param)
            results_df[string("param_", i)] = result.param[i]
        end
    catch myerror
        
        if throw_errors
            throw(myerror)
        end

        results_df = Dict{String, Any}(
            "obj_vals" => norm(result.resid),
            "opt_converged" => result.converged,
            "opt_runtime" => timeittook,
            "opt_error" => false,
            "opt_error_message" => sprint(showerror, myerror)
        )
    end

    # save to file
    if write_results_to_file == 2
        CSV.write(outputfile, DataFrame(results_df))
    end

    return results_df
end

"""

    gmm_2step()

Generalized method of moments (GMM) or classical minimum distance (CMD), with (optional) two-step optimal weighting matrix.

momfn = the moment function
    - should take a single vector argument
    - data etc is already "loaded" in this function
theta0 = initial condition (vector)
theta_lower, theta_upper = bounds (vectors, can include +/-Inf)
vcov_fn = function to compute variance covariance matrix. By default, vcov_gmm_iid which assumes data is iid.
n_theta0 = number of initial conditions
n_moms = size of moment function (number of moments) TODO: get this auto
results_dir_path = where to write results
Wstep1 = weighting matrix (default = identity matrix).
        Should be Hermitian (this will be enforced).
        Will be Cholesky decomposed
normalize_weight_matrix = boolean, if true, aim for the initial objective function to be <= O(1)
jacobian = provide jacobian function
write_results_to_file = 
    0 = nothing written to file
    1 = write objects to file, including one csv with all runs (once for first stage, once for second stage)
    2 = write objects to file, including one csv per each run
run_parallel  = individual runs in parallel vs in serial (default=parallel)
show_trace = show trace of curve_fit from LsqFit?
maxIter    = maximum iterations for curve_fit from LsqFit
show_progress = pass on to objective function
"""
function gmm_2step(;
			momfn_loaded,
			theta0,
			theta_lower,
			theta_upper,
            vcov_fn=nothing,
			run_parallel=true,
            two_step=false,
			n_moms=nothing,
			Wstep1=nothing,
            normalize_weight_matrix=false,
			results_dir_path="",
            write_results_to_file=0,
            overwrite_existing::String="all",
			maxIter=1000,
			maxTime=Inf,
            throw_errors::Bool=false,
            show_trace::Bool=false,
			show_theta::Bool=false,
            show_progress::Bool=false)

## Basic checks
    if write_results_to_file ∉ [0, 1, 2]
        error("write_results_to_file should be 0, 1, or 2")
    end

    if isnothing(vcov_fn)
        vcov_fn = vcov_gmm_iid
    end

## Store estimation results here
    gmm_results = Dict{String, Any}()

## if theta0 is a vector (one set of initial conditions), turn to 1 x n_params matrix
    # ? ever needed?
	if isa(theta0, Vector)
		theta0 = Matrix(transpose(theta0))
	end
    n_params = size(theta0, 2)
    n_theta0 = size(theta0, 1)

## if not provided, compute number of moments by running moment function once
	if isnothing(n_moms)
		n_moms = size(momfn_loaded(theta0[1,:]), 2)
	end

## save initial conditions to file
	theta0_df = DataFrame("iteration" => 1:n_theta0)
	for i=1:n_params
		theta0_df[!, string("param_", i, "_initial")] = vec(theta0[:, i])
	end

    # gmm_results["theta0_df"] = theta0_df
    # if write_results_to_file > 0
    #     outputfile = string(results_dir_path,"theta_initial_df.csv")
    #     CSV.write(outputfile, theta0_df)
    # end

## Initial weighting matrix W
	if isnothing(Wstep1)
        show_progress && println("GMM => using identity W1")

        # if not provided, use identity weighting matrix
        @assert isnothing(Wstep1)
		Wstep1 = diagm(ones(n_moms))
	end

	## save
    gmm_results["Wstep1"] = Wstep1
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"Wstep1.csv")
        CSV.write(outputfile, Tables.table(Wstep1), header=false)
    end

	# cholesky half. satisfies Whalf * transpose(Whalf) = W
	initialWhalf = Matrix(cholesky(Hermitian(Wstep1)).L)
	@assert norm(initialWhalf * transpose(initialWhalf) - Wstep1) < 1e-10

## normalize weight matrix such that the objective function is <= O(1) (very roughly speaking)
    if normalize_weight_matrix
        # initial guess = median along all initial conditions
		theta_initial = median(theta0, dims=1) |> vec

		# evaluable moments
		mom_matrix = momfn_loaded(theta_initial)

        # norm
        mom_norm = 1.0 + sqrt(norm(mean(mom_matrix, dims=1)))

        initialWhalf = initialWhalf .* det(initialWhalf)^(-1/n_moms) ./ mom_norm

        show_progress && println("GMM => Normalizing weight matrix.")
    end

## GMM first stage
    show_progress && println("GMM => Launching stage 1, number of initial conditions: ", n_theta0)

	# curve_fit in LsqFit requires to give data (x) to the objective function 
	# we hack this to give the GMM weighting matrix (Cholesky half)

	# optional: save results for each initial condition in a subdirectory
    results_subdir_path = string(results_dir_path, "step1/")
	if write_results_to_file == 2
		show_progress && print("GMM => Creating subdirectory to save one file per initial condition vector...")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# define objective function with moment function "loaded"
	gmm_obj_fn = (anyWhalf, any_theta) ->
					gmm_obj(theta=any_theta,
							Whalf=anyWhalf,
							momfn=momfn_loaded, # <- loaded moment function
							show_theta=show_theta)

    # run in parallel
	if run_parallel && n_theta0 > 1

	    all_results_df = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
                            overwrite_existing=overwrite_existing,
							maxIter=maxIter,
							maxTime=maxTime,
                            throw_errors=throw_errors,
                            show_trace=show_trace,
                            show_progress=show_progress
						), 1:n_theta0)
	else

        # not in parallel
		all_results_df = Vector{Any}(undef, n_theta0)
		for idx=1:n_theta0
	        all_results_df[idx] = curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
                            overwrite_existing=overwrite_existing,
							maxIter=maxIter,
							maxTime=maxTime,
                            throw_errors=throw_errors,
							show_trace=show_trace,
                            show_progress=show_progress)
		end
	end

    # DataFrame with all results
    all_results_df = vcat(DataFrame.(all_results_df)...)
    
	# pick smallest objective value (among those that have converged) 
    all_results_df.obj_vals_converged = copy(all_results_df.obj_vals)

    # TODO: allow iterations that have not converged (+ warning in docs)
    # if not converged, replace objective value with +Infinity
    all_results_df[.~all_results_df.opt_converged, :obj_vals_converged] .= Inf
	
    if minimum(all_results_df.obj_vals_converged) == Inf
        gmm_results["outcome_stage1"] = gmm_results["outcome"] = "fail"
        gmm_results["outcome_stage1_detail"] = ["none of the iterations converged"]

        empty_df = DataFrame()

        if write_results_to_file > 0
            open(results_dir_path * "est_results.json" ,"w") do f
                JSON.print(f, gmm_results, 4)
            end
        end

        return gmm_results, empty_df

    elseif any(.~all_results_df.opt_converged)

        n_converged = sum(all_results_df.opt_converged)

        gmm_results["outcome_stage1"] = gmm_results["outcome"] = "some_errors"
        gmm_results["outcome_stage1_detail"] = [string(n_converged) * "/" * string(n_theta0) * " iterations converged"]

        if minimum(all_results_df.obj_vals_converged) > minimum(all_results_df.obj_vals)
            push!(gmm_results["outcome_stage1_detail"], "minimum objective value occurs in iteration that did not converge")
        end

    else
        gmm_results["outcome_stage1"] = gmm_results["outcome"] = "success"
        gmm_results["outcome_stage1_detail"] = ["all iterations converged"]
    end
    
    # pick best
    idx_optimum = argmin(all_results_df.obj_vals_converged)
	all_results_df.is_optimum = ((1:n_theta0) .== idx_optimum)
	
    # select just the estimated parameters
    theta_hat_stage1 = gmm_results["theta_hat_stage1"] = all_results_df[idx_optimum, r"param_"] |> collect
	obj_val_stage1 = all_results_df[idx_optimum, :obj_vals]

	show_progress && println("GMM => Stage 1 optimal theta   ", theta_hat_stage1)
	show_progress && println("GMM => Stage 1 optimal obj val ", obj_val_stage1)

    # save
    # gmm_results["results_stage1"] = copy(all_results_df)
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"results_step1_df.csv")
	    CSV.write(outputfile, all_results_df)
    end 

	## if one-step -> stop here
    if ~two_step
        full_df = hcat(theta0_df, all_results_df)

        gmm_results["theta_hat"] = gmm_results["theta_hat_stage1"]

        if write_results_to_file > 0
            open(results_dir_path * "estimation_results.json" ,"w") do f
                JSON.print(f, gmm_results, 4)
            end
    
            CSV.write(results_dir_path * "estimation_results_df.csv", full_df)
        end

        return gmm_results, full_df
    end

## Optimal Weighting Matrix
    show_progress && println("GMM => Computing optimal weighting matrix")

    # by default, call vcov_gmm_iid() which computes the variance covariance matrix assuming data is iid
    # can also use user-provided function, useful when using classical minimum distance and non-iid data
	Wstep2 = vcov_fn(theta_hat_stage1, momfn_loaded)

    # ! sketchy
	# if super small determinant:
	# if det(Wstep2) < 1e-100
	# 	show_progress && println(" Matrix determinant very low: 1e-100. Adding 0.001 * I.")
	# 	Wstep2 = Wstep2 + 0.0001 * I
	# end

	# invert (still Hermitian)
	Wstep2 = inv(Wstep2)

    # save
    gmm_results["Wstep2"] = Wstep2
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"Wstep2.csv")
        CSV.write(outputfile, Tables.table(Wstep2), header=false)
    end

	# cholesky half. satisfies Whalf * transpose(Whalf) = W
	optimalWhalf = Matrix(cholesky(Wstep2).L)
	@assert norm(optimalWhalf * transpose(optimalWhalf) - Wstep2) < 1e-10

    ## normalize weight matrix such that the objective function is <= O(1) (very roughly speaking)
    if normalize_weight_matrix
        optimalWhalf = optimalWhalf .* det(optimalWhalf)^(-1/n_moms)

        show_progress && println("GMM => Normalizing weight matrix.")
    end

## GMM second stage
    show_progress && println("GMM => Launching stage 2, number of initial conditions: ", n_theta0)

	# optional: save results for each initial condition in a subdirectory
    results_subdir_path = string(results_dir_path, "step2/")
	if write_results_to_file == 2
		show_progress && print("GMM => Creating subdirectory to save one file per initial condition vector...")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# run in parallel
	if run_parallel && n_theta0 > 1

	    all_results2_df = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
							maxIter=maxIter,
							maxTime=maxTime,
                            throw_errors=throw_errors,
							show_trace=show_trace,
                            show_progress=show_progress), 1:n_theta0)
	else

        # not in parallel
		all_results2_df = Vector{Any}(undef, n_theta0)
		for idx=1:n_theta0
			all_results2_df[idx]=curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
							maxIter=maxIter,
							maxTime=maxTime,
                            throw_errors=throw_errors,
							show_trace=show_trace,
                            show_progress=show_progress)
		end
	end

    # one df with all results
    all_results2_df = vcat(DataFrame.(all_results2_df)...)

    # rename!(all_results2_df,
    #     :obj_vals => :obj_vals_step2,
    #     :opt_converged => :opt_converged_step2,
    #     :opt_runtime => :opt_runtime_step2)
    
    # for i=1:n_params
    #     rename!(all_results2_df, "param_" * string(i) => "param_" * string(i) * "_step2")
    # end

    # pick smallest objective value (among those that have converged) 
    all_results2_df.obj_vals_converged = copy(all_results2_df.obj_vals)

    # TODO: allow iterations that have not converged (+ warning in docs)
    # if not converged, replace objective value with +Infinity
    all_results2_df[.~all_results2_df.opt_converged, :obj_vals_converged] .= Inf

    if minimum(all_results2_df.obj_vals_converged) == Inf
        gmm_results["outcome_stage2"] = gmm_results["outcome"] = "fail"
        gmm_results["outcome_stage2_detail"] = ["none of the iterations converged"]

        empty_df = DataFrame()

        if write_results_to_file > 0
            open(results_dir_path * "estimation_results.json" ,"w") do f
                JSON.print(f, gmm_results, 4)
            end
        end

        return gmm_results, empty_df

    elseif any(.~all_results2_df.opt_converged)

        n_converged = sum(all_results2_df.opt_converged)

        gmm_results["outcome_stage2"] = "some_errors"
        gmm_results["outcome_stage2_detail"] =[string(n_converged) * "/" * string(n_theta0) * " iterations converged"]

        if minimum(all_results2_df.obj_vals_converged) > minimum(all_results2_df.obj_vals)
            push!(gmm_results["outcome_stage2_detail"], "minimum objective value occurs in iteration that did not converge")
        end

    else
        gmm_results["outcome_stage2"] = "success"
        gmm_results["outcome_stage2_detail"] = ["all iterations converged"]
    end

	# pick best
	idx_optimum = argmin(all_results2_df.obj_vals_converged)
	all_results2_df.is_optimum = ((1:n_theta0) .== idx_optimum)
	
    # select just the estimated parameters
    theta_hat_stage2 = gmm_results["theta_hat_stage2"] = all_results2_df[idx_optimum, r"param_[0-9]*"] |> collect
	obj_val_stage2 = all_results2_df[idx_optimum, :obj_vals]

    gmm_results["theta_hat"] = gmm_results["theta_hat_stage2"]

	show_progress && println("GMM => stage 2 optimal theta   ", theta_hat_stage2)
	show_progress && println("GMM => stage 2 optimal obj val ", obj_val_stage2)

    # save
    # gmm_results["results_stage2"] = copy(all_results2_df)
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"results_step2_df.csv")
	    CSV.write(outputfile, all_results2_df)
    end 

    # overall outcome of the GMM
    if (gmm_results["outcome_stage1"] == "some_errors") || (gmm_results["outcome_stage2"] == "some_errors")
        gmm_results["outcome"] = "some_errors"
    end

    full1_df = hcat(theta0_df, all_results_df)
    full2_df = hcat(theta0_df, all_results2_df)
    full1_df[!,:stage] .= 1
    full2_df[!,:stage] .= 2
    full_df = vcat(full1_df, full2_df)

    if write_results_to_file > 0
        open(results_dir_path * "estimation_results.json" ,"w") do f
            JSON.print(f, gmm_results, 4)
        end

        CSV.write(results_dir_path * "estimation_results_df.csv", full_df)
    end

	show_progress && println("GMM => complete")

    return gmm_results, full_df
end



## Serial (not parallel) bootstrap with multiple initial conditions

function bootstrap_2step(;
                    boot_run_idx,
					momfn,
					data,
					theta0_boot,
                    theta_lower,
                    theta_upper,
                    theta_hat, # so that we can evaluate the new moment at old parameters
                    sample_data_fn=nothing,
					boot_rng=nothing,
					# run_parallel=false,          # currently, always should be false

                    overwrite_existing::String="all",
					rootpath_boot_output,
					write_results_to_file=0,

					maxIter=100,
					maxTime=Inf,

                    show_trace=false,
					throw_errors=true,
                    show_theta=false,
                    show_progress=false
					)

	show_progress && print(".")

    # if result files already exist, do not run again:
    # boot_results_path = rootpath_boot_output * "estimation_results.json"
    # if ~overwrite_existing && isfile(boot_results_path)

    #     show_progress && println("\nSkipping boot iteration ", boot_run_idx) 

    #     # read results
    #     boot_result = JSON.parsefile(boot_results_path) 
        
    #     # df -- if it exists
    #     boot_results_df_path = rootpath_boot_output * "est_results_df.csv"
    #     if isfile(boot_results_df_path)
    #         boot_result_df = CSV.read(boot_results_df_path, DataFrame)
    #     else
    #         boot_result_df = DataFrame()
    #     end

    #     return boot_result, boot_result_df
    # end

    ## load data and prepare
    # TODO: do and document this better -- default = assume data is a dictionary of vectors/matrices
    # TODO: should also be able to use a boostrapping function
    # TODO: sample_data_fn(DATA::Any, boot_rng::RandomNumberSeed)

    if isnothing(sample_data_fn)
        data_dict_boot = copy(data)
        firstdatakey = first(sort(collect(keys(data_dict_boot))))
        n_observations = size(data[firstdatakey], 1)

        boot_sample = StatsBase.sample(boot_rng, 1:n_observations, n_observations)

        for mykey in keys(data_dict_boot)
            if length(size(data[mykey])) == 1
                data_dict_boot[mykey] = data[mykey][boot_sample]
            elseif length(size(data[mykey])) == 2
                data_dict_boot[mykey] = data[mykey][boot_sample, :]
            end
        end	
    else
        
        # apply the provided function
        data_dict_boot = sample_data_fn(data, boot_rng)
    end

## define the moment function with Boostrap Data
	momfn_loaded = theta -> momfn(theta, data_dict_boot)

## run 2-step GMM and save results
	boot_result, boot_result_df = gmm_2step(
			momfn_loaded=momfn_loaded,
			theta0=theta0_boot,
			theta_lower=theta_lower,
			theta_upper=theta_upper,
			run_parallel=false,
			results_dir_path=rootpath_boot_output,
			write_results_to_file=write_results_to_file, ## TODO: do what here?
            overwrite_existing=overwrite_existing,
			maxIter=maxIter,
			maxTime=maxTime,
            throw_errors=throw_errors,
            show_trace=show_trace,
            show_theta=show_theta,
			show_progress=false)

    # evaluate new boot moment function at the old parameter values
    boot_result["mom_at_theta_hat"] = momfn_loaded(theta_hat)

    # housekeeping
    boot_result["boot_run_idx"] = boot_run_idx
    boot_result_df[!, "boot_run_idx"] .= boot_run_idx

    # overwrite results and DF
    if write_results_to_file > 0
        open(rootpath_boot_output * "est_results.json" ,"w") do f
            JSON.print(f, boot_result, 4)
        end

        CSV.write(rootpath_boot_output * "est_results_df.csv", boot_result_df)
    end

    return boot_result, boot_result_df
# catch e
# 	println("BOOTSTRAP_EXCEPTION for ", rootpath_boot_output)

# 	bt = catch_backtrace()
# 	msg = sprint(showerror, e, bt)
# 	println(msg)

# 	if throw_exceptions
# 		throw(e)
#     else
#         boot_result = Dict{String, Any}(
#             "boot_run_idx" => boot_run_idx,
#             "outcome" => "fail",
#             "outcome_detail" => ["bootstrap error " * string(boot_run_idx)]
#         )

#         # TODO: add DF with all "failed"
        
# 	end
# end

end

function vector_theta_fix(mytheta, theta_fix)
    
    # indices of parameters to estimate
    idxs_estim = findall(isnothing, theta_fix)
    
    # different for vectors and matrices
    if isa(mytheta, Vector)
        return mytheta[idxs_estim]
    else
        return mytheta[:, idxs_estim]
    end

end


"""
Calls the moment function `momfn` combining `mytheta` and the fixed parameters in `theta_fix`
"""
function momfn_theta_fix(momfn, mytheta, mydata_dict, theta_fix)

    # indices of parameters that are fixed and those that need to be estimated
    idxs_fixed = findall(x -> ~isnothing(x), theta_fix)
    idxs_estim = findall(isnothing, theta_fix)

    vals_fixed = [theta_fix[idx] for idx=idxs_fixed]

    # make full parameter vector
    mytheta_new = zeros(length(theta_fix))
    mytheta_new[idxs_fixed] .= vals_fixed
    mytheta_new[idxs_estim] .= mytheta

    return momfn(mytheta_new, mydata_dict)
end

function omega_subset(myomega, moms_subset)
    if isa(myomega, Matrix)
        return myomega[moms_subset, moms_subset]
    elseif isa(myomega, Function)
        return (theta, momfn) -> myomega(theta, momfn)[moms_subset, moms_subset]
    end
end


"""
myfactor: higher factor = larger changes in parameters
max range -- in order to avoid sampling outside boundaries
"""
function compute_jacobian(;
        momfn,
        data,
        theta_hat,
        theta_upper=nothing, 
        theta_lower=nothing,
        theta_fix=nothing,
        moms_subset=nothing,
        myfactor=1.0)
    
    # subset parameter and moments, if applicable
    if ~isnothing(theta_fix)

        # define moment function taking as input only the variable entries in theta
        momfn2 = (mytheta, mydata_dict) -> momfn_theta_fix(momfn, mytheta, mydata_dict, theta_fix)
        
        # subset the other entries
        theta_hat   = vector_theta_fix(theta_hat, theta_fix)
        theta_upper = vector_theta_fix(theta_upper, theta_fix)
        theta_lower = vector_theta_fix(theta_lower, theta_fix)
            
    else
        momfn2 = momfn
    end

    if ~isnothing(moms_subset)
        # moms_subset is a sorted vector of distinct indices between 1 and n_moms 
        momfn3 = (mytheta, mydata_dict) -> momfn2(mytheta, mydata_dict)[:, moms_subset]
    else
        momfn3 = momfn2
    end
    
    # function that computes averaged moments
    mymomfunction_main_avg = mytheta -> mean(momfn3(mytheta, data), dims=1)

    # numerical jacobian
    my_max_range = 0.9 * min(minimum(abs.(theta_hat .- theta_lower)), minimum(abs.(theta_hat .- theta_upper)))

    # compute jacobian (from FiniteDifferences)
    myjac = jacobian(central_fdm(5, 1, factor=myfactor, max_range=my_max_range), mymomfunction_main_avg, theta_hat)

    return myjac[1]
end



function run_checks(;
        momfn,
        data,
        theta0, 
        theta_upper=nothing, 
        theta_lower=nothing,
        W=nothing,
        omega=nothing, # nothing, function or matrix
        theta_fix=nothing,
        moms_subset=nothing,
        gmm_options=nothing,
        theta0_boot=nothing,    # only add this when running from run_inference
    )

    ## make local copy and run checks
    gmm_options = copy(gmm_options)
    if ~isnothing(omega)
        omega = copy(omega)
    end

    if isnothing(omega) && (get(gmm_options, "estimator", "") == "cmd")
        error("If using CMD must provide omega = the variance-covariance matrix of the moments")
    end

    # TODO: to add?
    # runchecks(theta0, theta0_boot, theta_upper, theta_lower, gmm_options)

## Number of parameters
    # if only one initial condition as Vector, convert to 1 x n_params matrix
    if isa(theta0, Vector)
        theta0 = Matrix(transpose(theta0))
    end
    gmm_options["n_params"] = size(theta0)[2]

    # Default parameter bounds
    if isnothing(theta_lower) 
        theta_lower = fill(-Inf, n_params)
    end
    if isnothing(theta_upper) 
        theta_upper = fill(Inf, n_params)
    end

## Default options
	gmm_options_default = Dict(

        "param_names" => nothing, # vector of parameter names (strings)
        "n_observations" => nothing, # number of observations (data size)
        "n_moms" => nothing, # number of moments

        # one-step or two-step GMM
        "estimator" => "gmm2step", #"gmm1step" or "gmm2step" or "cmd" or "cmd_optimal"

        # main gmm estimation
        "normalize_weight_matrix" => false,

        "main_overwrite_existing" => "all", 
            # all=overwrite everything
            # "error"=overwrite only runs that errored 
            # "limit"=overwrite only runs that have hit the time or iterations limit
            # "error,limit"=overwrite both of the above
            # "none"=do not overwrite anything
		"main_write_results_to_file" => 0, # 0, 1 or 2, see definition in gmm_2step()

        "main_run_parallel" => false, # different starting conditions run in parallel
		"main_maxIter" 		=> 1000, # maximum number of iterations for curve_fit() from LsqFit.jl
		"main_maxTime" 	    => Inf,
        "main_throw_errors" => false, 
        "main_show_trace" 	=> false, # display optimization trace
		"main_show_theta" 	=> false, # during optimization, print current value of theta for each evaluation of momfn + value of objective function

        # inference:
        "var_asy" => true,  # Compute asymptotic variance covariance matrix
        "var_boot" => nothing,  # bootstrap. nothing or "quick" or "slow"

        # bootstrap:
        "boot_run_parallel" => false, # each bootstrap run in parallel.
		"boot_n_runs" 		=> 100, # number of bootstrap runs 
		
        "boot_overwrite_existing" => "all", 
            # all=overwrite everything
            # "error"=overwrite only runs that errored 
            # "limit"=overwrite only runs that have hit the time or iterations limit
            # "error,limit"=overwrite both of the above
            # "none"=do not overwrite anything
        "boot_write_results_to_file" => 0, # similar to "main_write_results_to_file"


		"boot_maxIter" 		=> 1000, 
		"boot_maxTime"	    => Inf, 
		"boot_throw_errors" => true, # if "false" will not crash when one bootstrap run has error. Error will be recorded in gmm_results["gmm_boot_results"]
        # ! add errors from boot in final results gmm_results["gmm_boot_results"]
        "boot_show_trace" 	=> false,
		"boot_show_theta"	=> false,

        # where to save results
		"rootpath_output" => "",

        # misc
        "show_progress" => true, # print overall progress/steps

        # use estimated parameters when the optimizer did not converge (due to iteration limit or time limit). 
        # Attention! Results may be wrong/misleading! Only use for testing or if this is OK for use case.
        "use_unconverged_results" => false
	)

    # for options that are not provided, use the default
    if isnothing(gmm_options) 
        gmm_options = Dict{String, Any}()
    end

	for mykey in keys(gmm_options_default)
		if ~haskey(gmm_options, mykey)
			gmm_options[mykey] = gmm_options_default[mykey]
		end
	end

## get number of observations in the data and number of moments
    if isnothing(gmm_options["n_observations"]) || isnothing(gmm_options["n_moms"])
        theta_test = theta0[1, :]
        mymoms = momfn(theta_test, data)
        gmm_options["n_observations"] = size(mymoms)[1]
        gmm_options["n_moms_full"] = size(mymoms)[2]
    end

## Fix some of the parameters (estimate only the others)
    if ~isnothing(theta_fix)

        # define moment function taking as input only the variable entries in theta
        momfn2 = (mytheta, mydata_dict) -> momfn_theta_fix(momfn, mytheta, mydata_dict, theta_fix)
        
        # subset the other entries
        theta0      = vector_theta_fix(theta0, theta_fix)
        theta_upper = vector_theta_fix(theta_upper, theta_fix)
        theta_lower = vector_theta_fix(theta_lower, theta_fix)
        isnothing(theta0_boot) || (theta0_boot = vector_theta_fix(theta0_boot, theta_fix))
            
    else
        momfn2 = momfn
    end

## Subset of moments?
    if ~isnothing(moms_subset)

        # basic checks: moms_subset is a sorted vector of distinct indices between 1 and n_moms 
        sort!(moms_subset)
        # TODO: add checks for moms_subset

        momfn3 = (mytheta, mydata_dict) -> momfn2(mytheta, mydata_dict)[:, moms_subset]

        gmm_options["n_moms"] = length(moms_subset)

        # update the variance-covariance matrix to the subset of moment we're using
        omega1 = omega_subset(omega, moms_subset)

    else
        momfn3 = momfn2
        gmm_options["n_moms"] = gmm_options["n_moms_full"]
        omega1 = omega
    end

## two-step estimator?
    gmm_options["2step"] = gmm_options["estimator"]  == "gmm2step"

## CMD
    if gmm_options["estimator"] == "cmd_optimal"
        # optimal W = Ω⁻¹
        W = inv(Symmetric(omega1))
        @assert issymmetric(W)

    elseif (gmm_options["estimator"] in ["cmd", "gmm1step"]) && isnothing(W)
        # optimal W = Ω⁻¹
        W = diagm(ones(gmm_options["n_moms"]))
        @assert issymmetric(W)

    elseif ~isnothing(W) && ~issymmetric(W)
        @warn "The provided weighting matrix W is not symmetric."
        W = Symmetric(W)
    end

## Number of initial conditions
    main_n_initial_cond = size(theta0, 1)

## Writing to file: if write main results, also write boot
    if gmm_options["main_write_results_to_file"] > 0
         gmm_options["boot_write_results_to_file"] = max(1, gmm_options["boot_write_results_to_file"])
    end    

    return gmm_options, theta0, theta_lower, theta_upper, theta0_boot, momfn3, omega1, W, main_n_initial_cond
end



"""
    run_estimation(; momfn, data, theta0, theta0_boot=nothing, theta_upper=nothing, theta_lower=nothing, gmm_options=nothing)

Wrapper for GMM/CMD with multiple initial conditions and optional bootstrap inference.

# Arguments
- momfn: the moment function momfn(theta, data)
- data: any object
- theta0: matrix of size main_n_start_pts x n_params, main_n_start_pts = gmm_options["main_n_start_pts] and n_params is length of theta
- theta0_boot: matrix of size (boot_n_start_pts * boot_n_runs) x n_params
- theta_lower: vector of lower bounds (default is -Inf)
- theta_upper: vector of upper bounds (default is +Inf)

Note: all arguments must be named (indicated by the ";" at the start), meaning calling [1] will work but [2] will not:
[1] run_estimation(momfn=my_moment_function, data=mydata, theta0=my_theta0)
[2] run_estimation(my_moment_function, mydata, my_theta0)
"""

function run_estimation(;
		momfn,
		data,
		theta0, 
        theta_upper=nothing, 
        theta_lower=nothing,
        W=nothing,
        omega=nothing, # nothing, function or matrix
        theta_fix=nothing,
        moms_subset=nothing,
        # sample_data_fn=nothing, # function for slow bootstrapping
		gmm_options=nothing
	)


## Run basic checks, fix parameters and subset moments (if applicable)
    gmm_options, theta0, theta_lower, theta_upper, theta0_boot, momfn3, omega1, W, main_n_initial_cond = run_checks(
            momfn=momfn,
            data=data,
            theta0=theta0,
            theta_upper=theta_upper,
            theta_lower=theta_lower,
            W=W,
            omega=omega,
            theta_fix=theta_fix,
            moms_subset=moms_subset,
            gmm_options=gmm_options)

## Store estimation options here
    estimation_parameters = Dict{String, Any}(
        "gmm_options"           => gmm_options,
        "theta0"                => theta0,      # initial conditions
        # "theta0_boot"         => theta0_boot,
        "theta_upper"           => theta_upper,
        "theta_lower"           => theta_lower,
        "W"                     => W,
        "omega"                 => omega,
        "theta_fix"             => theta_fix,
        "moms_subset"           => moms_subset,
        "n_observations"        => gmm_options["n_observations"],
        "n_moms"                => gmm_options["n_moms"],
        "n_moms_full"           => gmm_options["n_moms_full"],
        "n_params"              => gmm_options["n_params"],
        "main_n_initial_cond"   => main_n_initial_cond,
    )

    estimation_parameters_file = gmm_options["rootpath_output"] * "estimation_parameters.json"

    # if file already exists, check using same estimation options
    if (gmm_options["main_overwrite_existing"] != "all") && isfile(estimation_parameters_file)
        temp = JSON.parsefile(estimation_parameters_file)

        ## Debug if params files are not identical
        estimation_parameters_json = JSON.parse(JSON.json(estimation_parameters))

        # display(temp)
        # display(JSON.parse(JSON.json(estimation_parameters)))

        # for mykey in keys(temp)
        #     if temp[mykey] != estimation_parameters_json[mykey]
        #         println(mykey)
        #         display(temp[mykey])
        #         display(estimation_parameters_json[mykey])
        #         println()
        #     end
        # end

        # TODO: make sure that some key paramters are identical (initial conditions, etc.)
        # check_same_parameters()

        # if temp == estimation_parameters_json
        #     println("estimation_parameters.json file exists and identical to current parameters.")
        # else
        #     error("estimation_parameters.json file exists and differs from current parameters. Either select main_overwrite_existing=true or run with same parameters.")
        # end
    else

        # save 
        open(estimation_parameters_file ,"w") do f
            JSON.print(f, estimation_parameters, 4)
        end
    end
    
## Load data into moments function
	momfn_loaded = theta -> momfn3(theta, data)

## Run estimation: two-step GMM / CMD with optimal weighting matrix

    gmm_options["show_progress"] && println("Starting main estimation")

    # TODO: add "use_unconverged_results" option
    est_results, est_results_df  = gmm_2step(
        momfn_loaded    =momfn_loaded,

        theta0          =theta0,
        theta_lower     =theta_lower,
        theta_upper     =theta_upper,

        two_step        =gmm_options["2step"],
        Wstep1          =W,
        normalize_weight_matrix=gmm_options["normalize_weight_matrix"],
        vcov_fn         =omega,
        
        results_dir_path=gmm_options["rootpath_output"],
        write_results_to_file=gmm_options["main_write_results_to_file"],
        overwrite_existing=gmm_options["main_overwrite_existing"],
        
        run_parallel    =gmm_options["main_run_parallel"],
        maxIter         =gmm_options["main_maxIter"],
        maxTime         =float(gmm_options["main_maxTime"]),

        throw_errors    =gmm_options["main_throw_errors"],

        show_trace      =gmm_options["main_show_trace"],
        show_theta      =gmm_options["main_show_theta"],
        show_progress   =gmm_options["show_progress"])

### Collect results, clean up

    # return est_options, est_results, est_results_df
end






function run_inference(;
        momfn,
		data,
        theta0_boot=nothing,
        theta_upper=nothing, 
        theta_lower=nothing,
        W=nothing,
        omega=nothing, # nothing, function or matrix
        theta_fix=nothing,
        moms_subset=nothing,
        sample_data_fn=nothing, # function for slow bootstrapping
		gmm_options=nothing,
        
        est_results=nothing,     # dictionary with estimation results
        # est_results_path=nothing, # path to JSON with estimation results
        mt_seed=123
        )

## Get estimation results as a dictionary from JSON file
    if isnothing(est_results)
        est_results = JSON.parsefile(gmm_options["rootpath_output"] * "estimation_results.json")         
    end

## Run basic checks, fix parameters and subset moments (if applicable)
gmm_options, theta0, theta_lower, theta_upper, theta0_boot, momfn3, omega1, W = run_checks(
        momfn=momfn,
        data=data,
        theta0=theta0_boot,
        theta0_boot=theta0_boot,
        theta_upper=theta_upper,
        theta_lower=theta_lower,
        W=W,
        omega=omega,
        theta_fix=theta_fix,
        moms_subset=moms_subset,
        gmm_options=gmm_options)

## Store estimation results here
    # store parameters and options
    # est_results = Dict{String, Any}(
    #     "gmm_options" => gmm_options,
    #     "theta0" => theta0,
    #     "theta0_boot" => theta0_boot,
    #     "theta_upper" => theta_upper,
    #     "theta_lower" => theta_lower,
    #     "W" => W,
    #     "omega" => omega,
    #     "theta_fix" => theta_fix,
    #     "moms_subset" => moms_subset,
    #     "n_observations" => gmm_options["n_observations"],
    #     "n_moms" => gmm_options["n_moms"],
    #     "n_moms_full" => gmm_options["n_moms_full"],
    #     "n_params" => n_params,
    #     "main_n_initial_cond" => main_n_initial_cond,
    # )

    
## Misc
    
    boot_result_json = nothing
    boot_result_dfs  = nothing

## Load data into moments function
	momfn_loaded = theta -> momfn3(theta, data)


    ## Asymptotic Variance
    if (gmm_options["var_asy"] || gmm_options["var_boot"] == "quick") && est_results["outcome"] != "fail"

        gmm_options["show_progress"] && println("Computing asymptotic variance")

        # Get estimated parameter vector
        theta_hat = Float64.(est_results["theta_hat"])
        
        # function that computes averaged moments
        mymomfunction_main_avg = theta -> mean(momfn_loaded(theta), dims=1)

        # TODO: add autodiff option
        ## numerical jacobian
        # higher factor = larger changes in parameters
        myfactor = 1.0

        # max range -- in order to avoid sampling outside boundaries
        my_max_range = 0.9 * min(minimum(abs.(theta_hat .- theta_lower)), minimum(abs.(theta_hat .- theta_upper)))

        # compute jacobian
        myjac = jacobian(central_fdm(5, 1, factor=myfactor, max_range=my_max_range), mymomfunction_main_avg, theta_hat)

        G = myjac[1]

        # different formulas if optimal (2step GMM or optimal CMD) or any other weight matrix
        # same for 2-step optimal GMM and CMD
        # https://ocw.mit.edu/courses/14-386-new-econometric-methods-spring-2007/b8a285cadaa8203272ad3cbce3ef445f_ngmm07.pdf
        # https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/7ddedae5317fdd5424ff924688df7c7c_MIT14_384F13_rec12.pdf

        if gmm_options["estimator"] == "cmd"
            # (G'WG)⁻¹G' W Ω W G(G'WG)⁻¹
            # Ω = Symmetric(gmm_options["cmd_omega"])

            Ω = Symmetric(omega1)
            W = Symmetric(W)

            bread = inv(transpose(G) * W * G) 
            V = bread * transpose(G) * W * Ω * W * G * bread
        end

        if gmm_options["estimator"] == "cmd_optimal"
            # W = Ω⁻¹ so the above simplifies to (G'WG)⁻¹
            # Ω = Symetric(gmm_options["cmd_omega"])
            W = Symmetric(W)

            V = inv(transpose(G) * W * G) 
        end

        if gmm_options["estimator"] == "gmm1step"
            # (G'WG)⁻¹G' W Ω W G(G'WG)⁻¹

            if isnothing(omega1)
                vcov_fn = vcov_gmm_iid
            else
                vcov_fn = omega1
            end
            theta_hat_stage1 = est_results["theta_hat_stage1"]
            Ω = vcov_fn(theta_hat_stage1, momfn_loaded)

            W = Symmetric(W)

            bread = inv(transpose(G) * W * G) 
            V = bread * transpose(G) * W * Ω * W * G * bread
        end

        if gmm_options["estimator"] == "gmm2step"
            # W = Ω⁻¹ so the above simplifies to (G'WG)⁻¹
            W = Symmetric(est_results["Wstep2"])
            V = inv(transpose(G) * W * G) 
        end

        # treat GMM and CMD differently
        n_observations = gmm_options["n_observations"]

        est_results["G"] = G
        est_results["asy_vcov"] = V / n_observations
        est_results["asy_stderr"] = sqrt.(diag(V / n_observations))

        
        ### Quick bootstrap
        # https://schrimpf.github.io/GMMInference.jl/bootstrap/
        # https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/resources/mit14_382s17_lec3/
        # https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/resources/mit14_382s17_lec5/
        if gmm_options["var_boot"] == "quick"
            
            # √n(θ̂ -θ₀) ∼ (G'AG)⁻¹G'Aϵ where G is Jacobian and A is the weighting matrix
            M = (transpose(G) * W * G) \ (transpose(G) * W)

            # Z=g(X,θ̂ ), demeaned
            Z = momfn_loaded(theta_hat) .- mean(momfn_loaded(theta_hat), dims=1)

            boot_n_runs = gmm_options["boot_n_runs"]
            rng = MersenneTwister(123);
            boot_result_json = Vector(undef, boot_n_runs)
            for i=1:boot_n_runs
                boot_sample = StatsBase.sample(rng, 1:n_observations, n_observations)

                Z_boot = mean(Z[boot_sample, :], dims=1)

                theta_hat_boot = theta_hat + vec(M * transpose(Z_boot))

                boot_result_json[i] = Dict(
                    "theta_hat" => theta_hat_boot,
                    "boot_sample" => boot_sample
                )
            end
        end

        # write to file?
        if gmm_options["main_write_results_to_file"] > 0
            outputfile = string(gmm_options["rootpath_output"], "asy_vcov.csv")
            CSV.write(outputfile, Tables.table(est_results["asy_vcov"]), header=false)

            outputfile = string(gmm_options["rootpath_output"], "jacobian.csv")
            CSV.write(outputfile, Tables.table(est_results["G"]), header=false)
        end

    end
    # end

## Run "slow" bootstrap where we re-run the minimization each time    
    if gmm_options["var_boot"] == "slow"
        gmm_options["show_progress"] && println("Starting boostrap")

        if gmm_options["boot_write_results_to_file"] > 0
            rootpath_boot_output = gmm_options["rootpath_output"] * "boot/"
            isdir(rootpath_boot_output) || mkdir(rootpath_boot_output)
        end

        # Get estimated parameter vector
        theta_hat = Float64.(est_results["theta_hat"])

        # Define moment function for bootstrap -- target moment value at estimated theta
        mom_at_theta_hat = mean(momfn_loaded(theta_hat), dims=1)

        momfn_boot = (mytheta, mydata_dict) -> (momfn3(mytheta, mydata_dict) .- mom_at_theta_hat)

        # one random number generator per bootstrap run
        current_rng = MersenneTwister(mt_seed);
        # current_rng = MersenneTwister(123);
        boot_rngs = Vector{Any}(undef, gmm_options["boot_n_runs"])

        gmm_options["show_progress"] && println("Creating random number generator for boot run:")
        for i=1:gmm_options["boot_n_runs"]
            gmm_options["show_progress"] && print(".")
            
            # increment and update "current" random number generator
            boot_rngs[i] = Future.randjump(current_rng, big(10)^20)
            current_rng = boot_rngs[i]

            # can check that all these are different
            # print(boot_rngs[i])  
        end
        gmm_options["show_progress"] && println(".")

        # Todo: what are folder options for boot? (similar to main?)
        # Create folders where we save estimation results
        boot_folders = Vector{String}(undef, gmm_options["boot_n_runs"])
        for i=1:gmm_options["boot_n_runs"]
            boot_folders[i] = rootpath_boot_output * "boot_run_" * string(i) * "/"
            isdir(boot_folders[i]) || mkdir(boot_folders[i])
        end

        # Run bootstrap
        boot_n_runs = gmm_options["boot_n_runs"]

        gmm_options["show_progress"] && println("Bootstrap runs:")

        if gmm_options["boot_run_parallel"]
            boot_results = pmap(
            idx -> bootstrap_2step(
                        boot_run_idx=idx,
                        momfn=momfn_boot,
                        data=data,
                        theta0_boot=theta0_boot,
                        theta_lower=theta_lower,
                        theta_upper=theta_upper,
                        theta_hat=theta_hat,
                        sample_data_fn=sample_data_fn,
                        boot_rng=boot_rngs[idx],
                        
                        overwrite_existing=gmm_options["boot_overwrite_existing"],
                        write_results_to_file=gmm_options["boot_write_results_to_file"],
                        rootpath_boot_output=boot_folders[idx],
                        
                        maxIter=gmm_options["boot_maxIter"],
                        maxTime=float(gmm_options["boot_maxTime"]),

                        throw_errors=gmm_options["boot_throw_errors"],
                        show_trace=false,
                        show_theta=gmm_options["boot_show_theta"],
                        show_progress=gmm_options["show_progress"]
                    ), 1:boot_n_runs)
        else

            boot_results = Vector{Any}(undef, boot_n_runs)
            for boot_run_idx=1:boot_n_runs
                boot_results[boot_run_idx] = bootstrap_2step(
                        boot_run_idx=boot_run_idx,
                        momfn=momfn_boot,
                        data=data,
                        theta0_boot=theta0_boot,
                        theta_lower=theta_lower,
                        theta_upper=theta_upper,
                        theta_hat=theta_hat,
                        sample_data_fn=sample_data_fn,
                        boot_rng=boot_rngs[boot_run_idx],

                        overwrite_existing=gmm_options["boot_overwrite_existing"],
                        write_results_to_file=gmm_options["boot_write_results_to_file"],
                        rootpath_boot_output=boot_folders[boot_run_idx],

                        maxIter=gmm_options["boot_maxIter"],
                        maxTime=float(gmm_options["boot_maxTime"]),
                        throw_errors=gmm_options["boot_throw_errors"],
                        show_trace=false,
                        show_progress=gmm_options["show_progress"],
                        show_theta=gmm_options["boot_show_theta"]
                    )
            end
        end
        gmm_options["show_progress"] && println()

        # JSON and CSV with boot results
        boot_result_json = []
        boot_result_dfs = []
        for boot_result=boot_results
            mydict, mydf = boot_result
            mydict["boot_n_initial_cond"] = size(theta0_boot, 1)

            push!(boot_result_json, mydict)
            push!(boot_result_dfs, mydf)
        end

        boot_result_dfs = vcat(boot_result_dfs..., cols = :union)

        if gmm_options["boot_write_results_to_file"] > 0
            open(rootpath_boot_output * "results_boot.json" ,"w") do f
                JSON.print(f, boot_result_json, 4)
            end
    
            CSV.write(rootpath_boot_output * "results_boot_df.csv", boot_result_dfs)
        end

    end # end

    return boot_result_json, boot_result_dfs, gmm_options
end