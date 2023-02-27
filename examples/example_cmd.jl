using Pkg, Revise
#Pkg.activate(".")
using GMMTools

# ! for later: we added these packages to the GMMTools environment, but they are not technicall necessary
using GLM
using Random
using DataFrames
using Statistics


Random.seed!(1234)

# include("gmm_display.jl") # ! need this

## Generate data for testing. 
    # The model is a logit choice model over two driving routes (short and long), where utility is
    # a function of the time difference and any potential congestion charge on the "short" route
    # Utility is denominated in the currency (e.g. dollars)
    # Approx half of the agents are "treated" in an experiment where they face a fixed charge for using the short route.
    # The model parameters are alpha = value of travel time (in minutes) and sigma = logit variance parameter

    include("model_logit.jl")

    # true parameters (alpha, sigma)
    true_theta = [1.5, 10.0]

    rng = MersenneTwister(123);
    data_dict, model_params = generate_data_logit(N=50000, rng=rng)

## Define moments function with certain parameters already "loaded"

    # get data moments
    M, V = moms_data_cmd(data_dict)

    # model moments minus data moments
    # moments_gmm_loaded = (mytheta, mydata_dict) -> (moms_model_cmd(
    #     mytheta=mytheta, 
    #     mydata_dict=mydata_dict, 
    #     model_params=model_params) .- M)

    function moments_gmm_loaded(mytheta, mydata_dict)
        # sleep(0.001)

        if rand() < 0.001
            #error("cracra")
        end

        return moms_model_cmd(
            mytheta=mytheta, 
            mydata_dict=mydata_dict, 
            model_params=model_params) .- M
    end

    # Test
    theta0 = [1.5, 10.0]
    moments_gmm_loaded(theta0, data_dict)


## GMM options
    gmm_options = Dict{String, Any}(
        "main_run_parallel" => false,
        
        "estimator" => "cmd",

        "var_boot" => "slow",
        "boot_n_runs" => 5,

        "rootpath_output" => ".",#G:/My Drive/optnets/analysis/temp/",

        "main_write_results_to_file" => true,
        "boot_write_results_to_file" => true,

        "show_progress" => true,
        "boot_show_progress" => true,

        "main_overwrite_runs" => 2, ## 10=overwrite everything
        "boot_overwrite_runs" => 2, ## 10=overwrite everything

        "main_throw_errors" => true,
        "boot_throw_errors" => false
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
    s = run_estimation(
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