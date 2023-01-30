# GMMTools

[![Build Status](https://github.com/Gkreindler/GMMTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Gkreindler/GMMTools.jl/actions/workflows/CI.yml?query=branch%3Amain)


# Re-running interrupted or incomplete estimation cycle
If during an estimation cycle some runs are completed successfully while others raise errors or hit the time limit, re-running the entire estimation cycle is efficient, that is, it does not repeat those runs that finished successfully. When it takes a long time to complete an entire estimation cycle (e.g. because of many initial conditions, bootstrap, or both), this can be helpful. There are three possible types of issues that may arise:
1) The entire estimation cycle crashes, e.g. out of memory error, or cluster time limit reached, etc.
2) An individual run leads to an error.
3) An individual run exceeds `maxIter` or `maxTime`.

To enable this behavior, set in `gmm_options` the options `"main_overwrite_existing"` and `"boot_overwrite_existing"` to one of the following
```
"error"=overwrite only runs that errored 
"limit"=overwrite only runs that have hit the time or iterations limit
"error,limit"=overwrite both of the above
"none"=do not overwrite anything
```

## Understanding the file output from incomplete cycles
We save the results from each initial condition run in a separate file (one-row dataframe) in `"step1/results_df_run_<iter_n>.csv"`. After all runs are finished, we combine all results into a single dataframe in `"estimation_results_df.csv"`. (Analogous for two-step GMM.) To avoid a large number of files (thousands in the case of bootstrap with multiple initial conditions), we clean up and delete the individual run output files after the combined dataframe is generated.

Todo's
* delete individual dataframes after the combined dataframe is generated
* if combined dataframe exists, process it and only launch the runs that are not complete
