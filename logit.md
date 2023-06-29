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

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

