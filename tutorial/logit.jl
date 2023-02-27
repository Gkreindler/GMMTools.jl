# Logit example with GMM

#=
This is a tutorial ot understand estimation of decisions making processes with logit shocks using the GMMTools.jl package.

## Set up

The model is a logit choice model over two driving routes (short and long). Utility is denominated in the currency (e.g. dollars). Relative utility between routes is a function of how much longer the long route takes to travel as well as any tolls a driver faces in the route.

Approximately half of the agents are "treated" in an experiment where they face a fixed charge of using the shorter route.

Let $d_1$ be the short distance and $d_2$ be the long distance. Define $p$ as the price of the toll and $T_i$ whether an individual is treated. Define the utilities of the short and long distance, respectively as

$$
U_{1i} = \alpha d_{1i} +  T_i p + \sigma \epsilon_{1i}
U_{2i} = \alpha d_{2i} + \sigma \epsilon_{2i}
$$

In the model, the $\epsilon$ shocks are Type I extreme-valued with variance parameter $1$. A driver chooses the shorter route (Route 1), when $U_{1i} > U_{2i}$ As a consequence, we know the probability an individual chooses the shorter route is

$$
\mathbf{P}(\text{Shorter route}_i) = \mathbf{P}\left(\frac{-\alpha |d_{2i} - d_{1i}| + T_i p}{\sigma} > \epsilon_{12}\right)
$$

The unknown model parameters we will estimate are $\alpha$, the value of travel time, and $\sigma$ the logit variance parameter, are unknown and need to be estimated. $\epsilon_{12i} = \epsilon_{2i} - \epsilon_{1i}$ and is distributed according to the logistic distribution such that

$$
\epsilon_{12} \sim \frac{e^{\epsilon_{12}}}{1 + e^{\epsilon_{12}}}
$$

Threfore, the probability someone chooses the shorter route can be re-written as

$$
\mathbf{P}(\text{Shorter route}_i) = \frac{\exp\left(\frac{-\alpha |d_2 - d_1| + T_i p}{\sigma}\right)}{1 + \exp\left(\frac{-\alpha |d_2 - d_1| + T_i p}{\sigma}\right)}
$$

## Estimation Strategy

We are presented with a data with $N$ individuals, some are treated and some are not. We also observe the difference between their short and long routes.

As we are seeking to estimate two parameters, $\alpha$, the disutility of driving in nominal terms, and $\theta$, the variance of the preference shock, we will use two moments to guide our analysis. To show the strength of GMMTools, we will choose two moments which are somewhat opaque functions of the data directly. These will be the model-predicted mean of take-up in the treated and control groups respectively.


## Estimation strategy

Load the packages we will be using for the analysis
=#

using GMMTools
using Parameters
using UnPack
using GLM
using Random, Statistics

#=
Define containers for the known parameters in the model (the inputs) and the unknown parameters in the model (which we will determine via calibration)
=#

@with_kw struct KnownParams{T}
  p::T
end

#-

@with_kw struct UnKnownParams{T}
  α::T
  σ::T
end

#=
For a given individual facing a difference between the longer and shorter route `d > 0`, who may be treated or untreated, find the predicted probability of choosing the shorter route. We censor values to avoid numerical errors.
=#

function get_model_pred_takeup(
  dist,
  treated,
  kp::KnownParams,
  up::UnKnownParams;
  max_diff = 200)

  @unpack α, σ = up
  @unpack p = kp
  t = clamp((-α * dist + treated * p) / σ, -max_diff, max_diff)
  e = exp(t)
  e / (1 + e)
end

#=
When we have a *guess* of unknown parameters, we need to check how closely our model predictions match the observed choices. Here the data, which should be thought of as a `DataFrame`, `NamedTuple` of `Vector`s, or other object which supports `@unpack` access, contains the distances faced by each driver, whether they were treated, and whether or not they chose the shorter route. We treat the price of the toll as a known parameter.
We get moments from the observed data in a somewhat convoluted way, because the goal is to have a covariance matrix between the mean for the control group and the mean of the treated group. An eay way to get this is through linear regression.
=#

function get_moments_data(data)
  ## Get treated and control takeup means
  ## By omitting the intersection and including the (perfectly)
  ## collinear) vector of who is a control, coefficients represent
  ## group means.
  m = lm(@formula(takeups ~ 0 + controls + treatments), data)
  M = permutedims(coef(m))

  ## The covariance matrix is now easily estimated from the
  ## regression output
  V = vcov(m)

  M, V
end

#-

function get_moments_model(up::UnKnownParams, data, kp::KnownParams)
  @unpack treatments, distances, takeups = data
  treated_obs = treatments .== 1
  control_obs = treatments .== 0
  ## Get treated and control takeup means predicted by model
  pred_takeup = get_model_pred_takeup.(data.distances, data.treatments, Ref(kp), Ref(up))

  [mean(pred_takeup[control_obs]) mean(pred_takeup[treated_obs])]
end

#-

function get_moments(up::UnKnownParams, data, kp::KnownParams)
  M_data, V_data = get_moments_data(data)
  M_model = get_moments_model(up, data, kp)

  M_model .- M_data
end

#=
## Performing the estimation

In order to peform the estimation, first we need a way to generate data
=#

function generate_data(;
  N,
  max_distance,
  kp::KnownParams,
  true_up::UnKnownParams, # Actually known, here
  rng = nothing)

  distances = rand(rng, N) .* max_distance
  treatments = rand(rng, N).< .5
  controls = treatments .== 0

  takeup_probs = get_model_pred_takeup.(distances, treatments, Ref(kp), Ref(true_up))
  takeups = rand(rng, N) .< takeup_probs

  (; distances, treatments, controls, takeups)
end

#=
Now we finally use the GMMTools.jl library to estimate our unknown parameters. To do this we use the function `run_estimation`. `run_estimation` takes the following arguments:

- `momfn`: The moment function, used in the form `momfn(theta, data)` where `theta` is a vector of parameters to be estimated.
- `data`: The object passed to `momfn`
- `theta0`: Matrix representing starting guesses of `theta`. It is of size `main_n_start_pts` x `n_params` where `main_n_start_pts` is taken from the dictionary containing gmm options. `n_params` is the length of `theta`.
- `theta0_boot`: Akin to `theta0`, but contains starting values for each bootstrapping iteration. It is of size (`boot_n_start_pts` x `boot_n_runs`) x `n_params`
- `theta_lower`: Vector of lower bounds (default is -Inf)
- `theta_upper`: vector of upper bounds (default is +Inf)
=#

function estimate_gmm(data, kp::KnownParams; gmm_options = nothing)
  moment_fun = (θ, data) -> begin
    up = UnKnownParams(α = θ[1], σ = θ[2])
    get_moments(up, data, kp)
  end

  ## We will use the covariance matrix from the data in our
  ## weighting matrix
  _, V = get_moments_data(data)
  # Use the true moments we found above
  θ_0 = [1.5 10.0]
  run_estimation(;
    momfn = moment_fun,
    data = data,
    theta0 = θ_0,
    theta_lower = [0.0, 0.0],
    theta_upper =[Inf, Inf],
    omega = V,
    gmm_options)

end

#=
Running the set-up
=#
function main()
  rng = MersenneTwister(123)
  kp = KnownParams(p = 10.0)
  data = generate_data(;
    N = 50000,
    kp,
    max_distance = 20,
    true_up = UnKnownParams(α = 1.5, σ = 10.0),
    rng)

  gmm_options = Dict{String, Any}(
    "main_run_parallel" => false,

    "estimator" => "cmd",

    "var_boot" => "slow",
    "boot_n_runs" => 5,

    "rootpath_output" => ".",

    "main_write_results_to_file" => false,
    "boot_write_results_to_file" => false,
    "normalize_weight_matrix" => true,

    "show_progress" => true,
    "boot_show_progress" => true,

    "main_overwrite_runs" => 10, ## 10=overwrite everything
    "boot_overwrite_runs" => 10, ## 10=overwrite everything

    "main_throw_errors" => true,
    "boot_throw_errors" => false
  )

  x = estimate_gmm(data, kp; gmm_options)
end

#=
Now that we have a completed a toy example, let's take a look
at what is going on when we cann `estimate_gmm`

1. It checks all the inputs satisfy the criteria required.
2. It collects all the "estimation parameters". These are just options
   for reproducibility. Given that these are easy to calclate, they should
  be encoded in the code, not stored in a json.
3. Next it constructs an anonymous function which is "loaded" with the data.
   This is written without a `let` block so it has the danger of being boxed,
   but tbh I have no idea how real this is.

   I think the API should folor NLSolve and Optim and the user should just
   give the function directly.

4. Next it enters a function called `estimation_main` which is where the
   minimization actually occurs. It seems to take in the following keyword
   arguments.
    * `momfn_loaded`: The moment function
    * `theta0`: Initial guess
    * `theta_lower`: Lower bound of theta
    * `theta_upper`: Upper bound of theta
    * `two_step`: TODO: A two-step routine
    * `Wstep1`: The weighting matrix for the first step.
    * `normalize_weight_matrix`: TODO: Understand
    * `vcov_fn`: Variance-covariance. Not sure why it's called a function.
      I get that it's a function of the data, but it isn't always, and
      we didn't make it a function above.
    * `results_folder_path`: A path to save results
    * `write_results_to_file`: Whether to save results to a file
    * `overwrite_runs`: Overwrite estimated output. Uses some codes.
    * `run_parallel`: Run in parallel across processes
    * `maxIter`: Maximum number of iteratiorns passed to the fitting
    * `maxTime`: Maximum time passed to fitting
    * `throw_errors`
    * `show_trace`
    * `show_theta`
    * `show_progress`

It runs though a bunch of setup stuff. But a few notes
  * If there is no `vcov_fun` specified, it generates a function of the
    moment. From an Econometrics textbook we have
  $$
    \omega = \mathbb{E}(mm')
    \hat{\omega} = \frac{1}{N}\sum_{i = 1}^N mm'
  $$

  This matrix is generally symmetric (and because real, Hermitian).
  I'm not really sure why he chooses to wrap the output in Hermitian
  when he could just wrap in Symmetric. Maybe there are times when
  the result isn't real?

  * He saves the initial conditions in a file to keep track of.
  * He creates a Cholesky decomposition of the step 1 weighting
    matrix. It will be used later by `LsqFit`
  * He also normalizes the weight matrix such that the objective
    function is `<= O(1)`. This simply divides the Cholesky
    decomposition of the weighting matrix by the norm of the moments
    to easy computational burden.
  * OP uses `~` for negation which is really bad, since it's bitwise not
    and can apply to integers as well.

Next the code turns to the actual estimation routine, `estimation_one_step`.
  * First, it does a bunch of stuff related to paths and saving results.
  * It defines the objective function in an interesting way. Instead of writing

  $$
  m' W m
  $$

  it uses the Hermitian of W to write

  $$
  (m W^H)' (m W^H)
  $$

  which probably improves numerical performance and is something LsqFit.jl
  wants to have.

  * Remember that we have a lot of initial conditions. To optimize, it
    runs a lot of those initial conditions in a loop (which is
    distributed if need be).

Moving on to the curve fit wrapper.

  * Everything is wrapped in a `try` block, which may hurt performance.
    But a key feature of this package is to try candidate thetas which
    may throw errors.
  * Why are they not using the Jacobian? We pretty frequently know the
    Jacobian.
  *



=#