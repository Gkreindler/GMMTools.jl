# this script shows how to use "GMTools" to run estimation in parallel (over initial conditions and over bootstrap)

# using Pkg, Revise
# Pkg.activate(".")

using Distributed
n_procs = 4
if (n_procs > 1) && (n_procs > length(workers()))
	rmprocs(workers())
	display(workers())
	addprocs(n_procs)
	display(workers())
end

@everywhere begin

    using Pkg, Revise
    Pkg.activate(".")
end

@everywhere begin
    using GMMTools

    using FixedEffectModels
    using GLM
    using Random
    using DataFrames
    using Statistics

    Random.seed!(1234)
end

# include("gmm_display.jl") # ! need this

## Generate data for testing. 
    # The model is a logit choice model over two driving routes (short and long), where utility is a function of the time difference and any potential congestion charge on the "short" route
    # Utility is denominated in the currency (e.g. dollars)
    # Approx half of the agents are "treated" in an experiment where they face a fixed charge for using the short route.
    # The model parameters are alpha = value of travel time (in minutes) and sigma = logit variance parameter

    @everywhere include("model_logit.jl")

    # Note: by default, variables are created on the local (main) worker. they need to be explicitly created or copied to make them available on other workers
    # true parameters (alpha, sigma)
    true_theta = [1.5, 10.0]

    rng = MersenneTwister(123);
    data_dict, model_params = generate_data_logit(N=500, rng=rng)

    # make this data available on all workers (note the use of $ to reference a variable from the local worker)
    @everywhere data_dict = $data_dict
    @everywhere model_params = $model_params

## Define moments function with certain parameters already "loaded"

    # get data moments
    @everywhere M, V = moms_data_cmd(data_dict)

    # model moments minus data moments
    # moments_gmm_loaded = (mytheta, mydata_dict) -> (moms_model_cmd(
    #     mytheta=mytheta, 
    #     mydata_dict=mydata_dict, 
    #     model_params=model_params) .- M)

    @everywhere function moments_gmm_loaded(mytheta, mydata_dict)
        # sleep(0.001)

        # if rand() < 0.001
        #     error("cracra")
        # end

        return moms_model_cmd(
            mytheta=mytheta, 
            mydata_dict=mydata_dict, 
            model_params=model_params) .- M
    end

    # Test
    @everywhere moments_gmm_loaded([1.5, 10.0], data_dict)


## GMM options
    gmm_options = Dict{String, Any}(
        "main_run_parallel" => true,
        "boot_run_parallel" => true,
        
        "estimator" => "cmd",
        "cmd_omega" => V,  # variance-coveriance matrix

        "var_boot" => "slow",
        "boot_n_runs" => 5,

        "rootpath_output" => "G:/My Drive/optnets/analysis/temp/",

        "main_write_results_to_file" => true,
        "boot_write_results_to_file" => true,

        "show_progress" => true,
        "boot_show_progress" => true,

        "main_overwrite_runs" => 2, ## 10=overwrite everything
        "boot_overwrite_runs" => 2, ## 10=overwrite everything

        "main_throw_errors" => true,
        "boot_throw_errors" => true
    )

## Initial conditions (matrix for multiple initial runs) and parameter box constraints
    main_n_initial_cond = 20
    boot_n_initial_cond = 20

    theta_lower = [0.0, 0.0]
    theta_upper = [Inf, Inf]

    theta0      = random_initial_conditions([1.0 5.0], theta_lower, theta_upper, main_n_initial_cond)
    theta0_boot = random_initial_conditions([1.0 5.0], theta_lower, theta_upper, boot_n_initial_cond)

## Run GMM

    # est_options2, est_results, est_results_df = 
    run_estimation(
        momfn=moments_gmm_loaded,
        data=data_dict,
        theta0=theta0,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        omega=V,
        gmm_options=gmm_options)

    run_inference(
        momfn=moments_gmm_loaded,
        data=data_dict,
        theta0_boot=theta0_boot,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        omega=V,
        gmm_options=gmm_options)

## print model_results
    # print_results(gmm_results, results_df)