# Estimating a logit decision model

This tutorial demonstrates how to use GMMTools.jl to estimate preference parameters in a simple binary choice model with a logit structure. 

## Set up

A driver needs to go to work, and needs to choose between a short route with a toll and a longer, but free, route.
The driver's utility is denominated in the currency (e.g. dollars). Relative utility between routes is a function of how much longer the long route takes to travel as well as any tolls a driver faces in the route. 

We observe the decisions of many drivers. Our goal is to use this data to estimate underlying preference parameters of the driving population at large. 

We use a field experiment helps us with this estimation. Approximately half of the agents are "treated" in an experiment where they face a fixed charge of using the shorter route. For the control group, both routes are free. Comparing the number of drivers who take the shorter route, conditional on distance, in treated and control groups teaches us about underlying preferences. 

Let $d_1$ be the short distance and $d_2$ be the long distance. Define $p$ as the price of the toll and $T_i$ whether an individual is treated. Define the utilities of the short and long distance, respectively as

```math
U_{1i} = \alpha d_{1i} +  T_i p + \sigma \epsilon_{1i} \\
U_{2i} = \alpha d_{2i} + \sigma \epsilon_{2i}
```

In the model, the $\epsilon$ shocks are Type I extreme-valued with variance parameter $1$. A driver chooses the shorter route (Route 1), when $U_{1i} > U_{2i}$ As a consequence of our assumptions, we know the probability an individual chooses the shorter route is

```math
\mathbf{P}(\text{Shorter route}_i) = \mathbf{P}\left(\frac{-\alpha |d_{2i} - d_{1i}| + T_i p}{\sigma} > \epsilon_{2i} - \epsilon_{1i}\right)
```

Define $\epsilon_{12_i} = \epsilon_{2i} - \epsilon_{1i}$, which is distributed according to the logistic distribution such that

```math
\epsilon_{12} \sim \frac{e^{\epsilon_{12}}}{1 + e^{\epsilon_{12}}}
```

Define $d$ as the difference between the long and short route, $d = |d_2 - d_1|$. Therefore, the probability someone chooses the shorter route can be re-written as

```math
\mathbf{P}(\text{Shorter route}_i \mid d_i, T_i) = \frac{\exp\left(\frac{-\alpha d_i + T_i p}{\sigma}\right)}{1 + \exp\left(\frac{-\alpha d + T_i p}{\sigma}\right)}
```

The unknown model parameters we will estimate are $\alpha$, the value of travel time, and $\sigma$ the logit variance parameter. 

## Estimation Strategy

We are presented with a data with $N$ individuals, half are treated and half are not. We  observe the time and cost difference between their short and long routes as well as their choices. Our goal is to guess $\theta = (\alpha, \sigma)$ such that a theoretical prediction of drivers' choices best matches our observed data. 


To show the strength of GMMTools, we will choose two moments which are  opaque functions of the data directly. These will be the model-predicted mean of take-up in the treated and control groups respectively. Define $g$ to be the expected values of take-up for the control and treated groups according to the model described above. 

```math
g(\theta) = \begin{bmatrix}
\frac{1}{N/2}\sum_{i \mid T_i = 0} \mathbf{P}(\text{Shorter route}_i \mid d_i, T_i)   \\
\frac{1}{N/2}\sum_{i \mid T_i = 0} \mathbf{P}(\text{Shorter route}_i \mid d_i, T_i) 
\end{bmatrix} 
```

```math
\hat{g} = \begin{bmatrix}
\frac{1}{N_0}\sum_{i \mid T_i = 0} \text{Shorter route}_i  \\
\frac{1}{N_1}\sum_{i \mid T_i = 1} \text{Shorter route}_i
\end{bmatrix}
```

Define the anal

We select $\hat{\theta}$ to minimize 
```math
|| g(\hat{\theta}) - \hat{g}||
```

## Setting up the estimation

Load the packages we will be using for the analysis

```julia
using GMMTools
using Parameters
using UnPack
using GLM
using Random, Statistics, LinearAlgebra
```

Define containers for the known parameters in the model (the inputs) and the unknown parameters in the model (which we will determine via calibration)

```julia
@with_kw struct KnownParams{T}
  p::T
end
```

```julia
@with_kw struct UnKnownParams{T}
  α::T
  σ::T
end
```

For a single individual, find $\mathbf{P}(\text{Shorter route}_i \mid d_i, T_i)$. We censor values to avoid numerical errors.

```julia
function get_indiv_pred_takeup(
  dist,
  treated,
  kp::KnownParams,
  up::UnKnownParams;
  max_diff = 200)

  @unpack α, σ = up
  @unpack p = kp
  inside_exp = (-α * dist + treated * p) / σ
  t = clamp(inside_exp, -max_diff, max_diff)
  e = exp(t)
  e / (1 + e)
end
```

We use the function above to now calculate $g(\hat{\theta})$. We return a $1\times 2$ matrix, in accordance with the expectations of the GMMTools.jl API.   

```julia
function get_moments_model(up::UnKnownParams, data, kp::KnownParams)
  @unpack treatments, distances, takeups = data
  treated_obs = treatments .== 1
  control_obs = treatments .== 0
  # Get treated and control takeup means predicted by model
  pred_takeup = get_indiv_pred_takeup.(data.distances, data.treatments, Ref(kp), Ref(up))

  [mean(pred_takeup[control_obs]) mean(pred_takeup[treated_obs])]
end
```


In the third step, we calculate $\hat{g}$, the observed means of take-up between treated and control groups. We also return the variance-covariance matrix of $\hat{g}$, which we will use later in estimation. 

```julia
function get_moments_data(data)
  # Get treated and control takeup means
  # By omitting the intersection and including the (perfectly)
  # collinear) vector of who is a control, coefficients represent
  # group means.
  m = lm(@formula(takeups ~ 0 + controls + treatments), data)
  M = permutedims(coef(m))

  # The covariance matrix is now easily estimated from the
  # regression output
  V = vcov(m)

  M, V
end
```

Putting the above together, we estimate 

```math
||g(\hat{\theta}) - \hat{g} ||
```

```julia
function get_moments_diff(up::UnKnownParams, data, kp::KnownParams)
  M_data, V_data = get_moments_data(data)
  M_model = get_moments_model(up, data, kp)

  M_model .- M_data
end
```

Finally, we define a small function to generate the data so that we can perform the estimation. We input a random number generator to this function for reproducibility. 

```julia
function generate_data(;
  N,
  max_distance,
  kp::KnownParams,
  true_up::UnKnownParams, # Actually known, here
  rng = nothing)

  distances = rand(rng, N) .* max_distance
  treatments = rand(rng, N).< .5
  controls = treatments .== 0

  takeup_probs = get_indiv_pred_takeup.(distances, treatments, Ref(kp), Ref(true_up))
  takeups = rand(rng, N) .< takeup_probs

  (; distances, treatments, controls, takeups)
end
```


## Performing the estimation

First, generate data according to the true parameters. 

```
rng = MersenneTwister(123)

kp = KnownParams(p = 10.0)

true_up = UnKnownParams(α = 1.5, σ = 10.0)

data = generate_data(;
  N = 50000,
  kp,
  max_distance = 20,
  true_up, 
  rng)
```


Now we finally use the GMMTools.jl library to estimate our unknown parameters. To do this we use the function `run_estimation`. `run_estimation` takes the following arguments:

- `momfn`: The moment function, used in the form `momfn(theta, data)` where `theta` is a vector of parameters to be estimated.
- `data`: The object passed to `momfn`
- `theta0`: Matrix representing starting guesses of `theta`. It is of size `main_n_start_pts` x `n_params` where `main_n_start_pts` is taken from the dictionary containing gmm options. `n_params` is the length of `theta`.
- `theta0_boot`: Akin to `theta0`, but contains starting values for each bootstrapping iteration. It is of size (`boot_n_start_pts` x `boot_n_runs`) x `n_params`
- `theta_lower`: Vector of lower bounds (default is -Inf)
- `theta_upper`: vector of upper bounds (default is +Inf)


First, we define `momfn` according to the requirements above

```
moment_fun = let kp = kp # Capture the known parameters
  (θ, data) -> begin
    up = UnKnownParams(α = θ[1], σ = θ[2])
    get_moments_diff(up, data, kp)
  end
end
```

Confirm that the moment function returns the correct $1 \times 2$ matrix

```julia
moment_fun([1, 2], data)
```

Next, we define the GMM options that will control details of our estimation

```
output_dir = mktempdir()
gmm_options = Dict{String, Any}(
  # Do not run the estimation in parallel
  "main_run_parallel" => false,

  # Use a minimum distance estimator, rather than a 
  # GMM estimator. 
  "estimator" => "cmd",

  # Re-run the estimation when we calculate standard
  # errors, rather than re-use original parameter 
  # estimates
  "var_boot" => "slow",
  "boot_n_runs" => 5,

  # Folder to store intermediate output 
  "rootpath_output" => output_dir,  

  # Write intermediate parameter estimation
  # to a file
  "main_write_results_to_file" => true,

  # Do not write the bootstrapping to file
  "boot_write_results_to_file" => false,

  # Print out progress in estimation of θ
  "show_progress" => true,
  
  # Do not show progress for bootstrapping
  "boot_show_progress" => false,

  # Overwrite results
  "main_overwrite_runs" => 10, ## 10=overwrite everything
  "boot_overwrite_runs" => 10, ## 10=overwrite everything

  # Throw errors during estimation of θ
  "main_throw_errors" => true,

  # Do not throw errors during bootstrapping
  "boot_throw_errors" => false
)
```

Finally we are ready to run the estimation using the `run_estimation` function from GMMTools.jl

```julia
# Initial guess of θ
θ_0 = [1.5 10.0]
res = run_estimation(;
  momfn = moment_fun,
  data = data,
  theta0 = θ_0,
  theta_lower = [0.0, 0.0],
  theta_upper = [Inf, Inf],
  # Use the identity matrix to weight
  omega = I,
  gmm_options)
```

## Inspecting the output

The output `res` gives two objects, a `DataFrame` storing the minimum distance, best parameters, etc. and a `Dict` showing errors which may have occurred in the estimation.

```julia
res_df, res_dict = res
```

Inspecting `res_df.param_1` and `res_df.param_2`, we get the values 

```math
\hat{\alpha} = 1.510 \\
\hat{\sigma} = 9.972
```

The optimizer was able to get close to the true values of $\alpha = 1.5$ and $\sigma = 10$. 

Inspecting `res_dict` shows there were no errors and no non-convergences during estimation. 

These the results that exist in the Julia session. But GMMTools.jl also saves results to files. Let's inspect these individually. They are stored in `output_dir`, which we passed to the `gmm_options` dictionary above.

```
files = readdir(output_dir)
readshow(f) = println(read(joinpath(output_dir, f), String))
```

```
3-element Vector{String}:
 "estimation_flags.json"
 "estimation_parameters.json"
 "estimation_results_df.csv"
```

```julia
readshow("estimation_flags.json")
```

```
{
    "n_success": 1,
    "n_errors": 0,
    "n_not_converged": 0
}
```

This corresponds to `res_d`

```julia
readshow("estimation_parameters.json")
```

```
{
    "gmm_options": {
        "rootpath_output": "/tmp/jl_I6JkzR",
        "use_unconverged_results": false,
        "var_boot": "slow",
        "boot_maxIter": 1000,
        "main_show_theta": false,
        "main_throw_errors": true,
        "main_write_results_to_file": true,
        "boot_overwrite_runs": 10,
        "main_maxIter": 1000,
        "boot_throw_errors": false,
        "estimator": "cmd",
        "n_params": 2,
        "main_maxTime": null,
        "n_moms_full": 2,
        "boot_maxTime": null,
        "boot_show_progress": false,
        "main_show_trace": false,
        "boot_n_runs": 5,
        "boot_write_results_to_file": true,
        "show_progress": true,
        "var_asy": true,
        "boot_run_parallel": false,
        "2step": false,
        "main_n_initial_cond": 1,
        "normalize_weight_matrix": false,
        "boot_show_theta": false,
        "main_run_parallel": false,
        "boot_show_trace": false,
        "main_overwrite_runs": 10,
        "n_moms": 2,
        "n_observations": 1,
        "param_names": null
    },
    "W": [
        [
            1.0,
            0.0
        ],
        [
            0.0,
            1.0
        ]
    ],
    "theta0": [
        [
            1.5
        ],
        [
            10.0
        ]
    ],
    "main_n_initial_cond": 1,
    "theta_fix": null,
    "n_moms": 2,
    "n_params": 2,
    "theta_upper": [
        null,
        null
    ],
    "theta_lower": [
        0.0,
        0.0
    ],
    "n_moms_full": 2,
    "moms_subset": null,
    "omega": {
        "λ": true
    },
    "n_observations": 1
}
```

This corresponds to the full set of inputs to the key function `run_estimation`. Reading this file back in will let us fully reproduce our original estimation procedure. 

```julia
readshow("estimation_results_df.csv")
```

```
obj_vals,opt_converged,opt_error,opt_error_message,opt_runtime,param_1,param_2,run_idx,is_optimum
1.277058133722624e-15,true,false,,0.498569576,1.5101382430140904,9.971757744026124,1,true
```

This corresponds to `res_df`, which holds the parameter estimates. 

## Fine-tuning the optimization

### Using GMM estimation

Here we perform a one-step GMM estimation, equivalent to solving

```math
\min_{\theta} (\hat{g} - g(\hat{\theta}))' \Omega^{-1} (\hat{g} - g(\hat{\theta})
```

Where $\Omega$ represents the variance-covariance matrix of $\hat{g} - g(\hat{\theta})$. For our purposes, we use the variance-covariance matrix of $\hat{g}$, calculated with `get_moments_data`

```julia
_, V = get_moments_data(data)

gmm_options_one_step = let 
  t = copy(gmm_options)
  t["estimator"] = "gmm1step"
  t
end

res_one_step = run_estimation(;
  momfn = moment_fun,
  data = data,
  theta0 = θ_0,
  theta_lower = [0.0, 0.0],
  theta_upper = [Inf, Inf],
  # Use the identity matrix to weight
  omega = V,
  gmm_options = gmm_options_one_step)

res_one_step_df, res_one_step_d = res_one_step
```

### Multiple initial conditions

Often, the function we are minimizing has many local minima. To ensure we are finding the global minimum, is is helpful to use random initial conditions. We do this with the `random_initial_conditions` function. 



```julia
θ_0 = [1.5 10.0]

θ_0s_random = random_initial_conditions(θ_0, [0, 0], [Inf, Inf], 100)

res_random = run_estimation(;
  momfn = moment_fun,
  data = data,
  theta0 = θ_0s_random,
  theta_lower = [0.0, 0.0],
  theta_upper = [Inf, Inf],
  # Use the identity matrix to weight
  omega = I,
  gmm_options)

res_random_df, res_random_d = res_random
```


### Running in parallel with multiple initial conditions

When we optimize with various initial conditions, these processes are entirely independent and can thus occur in parallel. 

```julia
θ_0 = [1.5 10.0]

θ_0s_random = random_initial_conditions(θ_0, [0, 0], [Inf, Inf], 100)

gmm_options_parallel = let 
  t = copy(gmm_options)
  t["main_run_parallel"] = true
  t
end

res_parallel = run_estimation(;
  momfn = moment_fun,
  data = data,
  theta0 = θ_0s_random,
  theta_lower = [0.0, 0.0],
  theta_upper = [Inf, Inf],
  # Use the identity matrix to weight
  omega = I,
  gmm_options)

res_parallel_df, res_parallel_d = res_parallel
```









