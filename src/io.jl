


function Base.show(io::IO, r::GMMFit)
    println("GMMFit object. Dictionary version below:")

    temp_dict = to_dict(r)
    display(temp_dict)
end

function Base.show(io::IO, r::GMMvcov)
    println("GMMvcov (Variance-covariance) object. Dictionary version below:")

    temp_dict = to_dict(r)
    (:boot_fits in keys(temp_dict)) && (temp_dict[:boot_fits] = "GMMBootFit object")
    display(temp_dict)
end

function Base.show(io::IO, r::GMMBootFits)
    println("GMMBootFits object. Dictionary version below:")

    temp_dict = to_dict(r)
    (:boot_fits in keys(temp_dict)) && (temp_dict[:boot_fits] = "vector of GMMFit objects")
    display(temp_dict)
end

### Writing and Reading GMMFit objects

to_dict(myfit::Union{GMMFit, GMMvcov, GMMBootFits}) = Dict(k => getfield(myfit, k) for k ∈ fieldnames(typeof(myfit)))
to_dict(boot_fits::Vector{GMMFit}) = [to_dict(boot_fits[i]) for i=1:length(boot_fits)]


function parse_json(myfit::Union{GMMFit, GMMvcov})

    # convert to dictionary
    myfit_dict = to_dict(myfit)

    # write as JSON
    return JSON.json(myfit_dict, 2)
end



"""
Write GMMFit object to files: JSON for most fields + CSV for fits_df table
All paths should exist.
"""
function write(myfit::GMMFit, full_path; subpath="fit")
    
    # paths
    if full_path == "" 
        return
    end    
    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath

    myfit = deepcopy(myfit)

    # write vcov, then remove
        if !isnothing(myfit.vcov)
            write(myfit.vcov, opts)
            myfit.vcov = nothing
        end
    
    # write table(s), then remove
        if isnothing(myfit.fits_df)
            myfit.fits_df = table_fit(myfit)
        end
        CSV.write(full_path * ".csv", myfit.fits_df)
        myfit.fits_df = nothing

        # moments at theta_hat (if computed)
        if !isnothing(myfit.moms_hat)
            writedlm(full_path * "_moms_hat.csv", myfit.moms_hat, ',')
            myfit.moms_hat = nothing
        end

    # write JSON file (in the process, parse GMMFit object and var-covar to dict)
        fpath = full_path * ".json"
        open(fpath, "w") do file
            Base.write(file, parse_json(myfit))
        end
    
    return
end

### Variance-covariance results

function write(myvcov::GMMvcov, full_path; subpath="vcov")
    
    if full_path == ""
        return
    end
    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath

    myvcov = deepcopy(myvcov)

    # deal with tables
    if myvcov.method == :bayesian_bootstrap

        writedlm(full_path * "_boot_weights.csv", myvcov.boot_fits.boot_weights, ',')
        CSV.write(full_path * "_boot_fits_df.csv", myvcov.boot_fits.boot_fits_df)
        CSV.write(full_path * "_boot_moms_hat_df.csv", myvcov.boot_fits.boot_moms_hat_df)        

        # boot fit objects (vector of GMMFit objects)
        boot_fits_dict = to_dict(myvcov.boot_fits.boot_fits)

        # overwrite the bootstrap object
        myvcov.boot_fits_dict = boot_fits_dict
        myvcov.boot_fits = nothing
    end

    # the rest
    fpath = full_path * ".json"
    open(fpath, "w") do file
        Base.write(file, parse_json(myvcov))
    end

    return
end

"""
Delete intermediate files
"""
function clean_iter(opts)
    try
        if isdir(opts.path * "__iter__/")
            (opts.trace > 0) && print("Deleting intermediate files from: ", opts.path)
            rm(opts.path * "__iter__/", force=true, recursive=true)
            (opts.trace > 0) && println(" Done.")
        else
            (opts.trace > 0) && println("No intermediate files to delete.")
        end
    catch e
        println(" Error while deleting intermediate files from : ", opts.path, ". Error: ", e)
    end
end



### Reading GMMFit objects

# function parse_vector(s::AbstractString)
#     return parse.(Float64, split(s[2:(end-1)],","))
# end

function parse_weight_matrix(W)
    if isa(W, Dict) && ("λ" ∈ keys(W))
        return I
    else

        # JSON writes the matrix as a vector of vectors
        return convert.(Float64, stack(W))
    end
end

"""
replace nothing with missing (this happens when the fit has errored)
"""
function parse_vector(myvec)
    mysample = isnothing.(myvec)
    myvec[mysample] .= missing
    myvec[.!mysample] .= convert.(Float64, myvec[.!mysample])
    return myvec
end

function nothing2missing(x; mytype=Float64)
    return isnothing(x) ? missing : convert(mytype, x)
end

function nothing_or_convert(x; mytype=Float64)
    return isnothing(x) ? nothing : convert(mytype, x)
end

function dict2fit(myfit_dict)

    # JSON.print(myfit_dict, 2)

    if isnothing(myfit_dict["weights"])
        weights = nothing
    else
        weights = convert.(Float64, myfit_dict["weights"])
    end

    # handle fits that errored
    isnothing(myfit_dict["obj_value"]) && (myfit_dict["obj_value"] = Inf)
    isnothing(myfit_dict["converged"]) && (myfit_dict["converged"] = false)

    myfit = GMMFit(
            theta0          =nothing2missing.(myfit_dict["theta0"]),
            theta_hat       =nothing2missing.(myfit_dict["theta_hat"]),
            theta_names     =myfit_dict["theta_names"],
            theta_factors   =nothing_or_convert.(myfit_dict["theta_factors"]), 

            moms_hat        = myfit_dict["moms_hat"],
            n_obs           =myfit_dict["n_obs"],
            n_moms          =myfit_dict["n_moms"],

            mode            =Symbol(myfit_dict["mode"]),
            weights         =weights,
            W               =parse_weight_matrix(myfit_dict["W"]),

            obj_value       =convert.(Float64, myfit_dict["obj_value"]),
            converged       =myfit_dict["converged"],
            iterations      =nothing2missing(myfit_dict["iterations"], mytype=Int64),
            iteration_limit_reached = nothing2missing(myfit_dict["iteration_limit_reached"], mytype=Bool),
            time_it_took    =nothing2missing(myfit_dict["time_it_took"], mytype=Float64),

            idx             =myfit_dict["idx"])

    return myfit
end

"""
Example: fit object saved under "C:/temp/fit.json" and "C:/temp/fit.csv". Then call `read_fit("C:/temp/")`.
"""
function read_fit(full_path; subpath="fit", show_trace=false)

    (full_path == "") && return nothing

    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath
    
    # files exist?
    files_exist = isfile(full_path * ".csv") && isfile(full_path * ".json")

    if !files_exist
        show_trace && println("files `fit.csv` and/or `fit.json` do not exist in ", full_path)
        return nothing
    end

    # read and parse JSON file
    myfit_dict = JSON.parsefile(full_path * ".json")
    myfit = dict2fit(myfit_dict)

    # read CSV file with estimates
    df = CSV.read(full_path * ".csv", DataFrame)
    myfit.fits_df = df

    # read CSV file with moments at theta_hat
    if isfile(full_path * "_moms_hat.csv")
        myfit.moms_hat = readdlm(full_path * "_moms_hat.csv", ',', Float64)
    end

    @info "Read fit from file from " * full_path 

    # try to automatically read vcov object (myfit.vcov = nothing if this fails)
    myfit.vcov = read_vcov(full_path, show_trace=show_trace)

    return myfit
end

"""
Example: vcov object saved under "C:/temp/vcov.json". Then call `read_vcov("C:/temp/")`.
"""
function read_vcov(full_path; subpath="vcov", show_trace=false)
    
    (full_path == "") && return nothing

    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath

    # files exits?
    if !isfile(full_path * ".json")
        show_trace && println("file `vcov.json` does not exist in ", full_path)
        return nothing
    end

    # read JSON file
    myvcov_dict = JSON.parsefile(full_path * ".json")

    # fix matrices
    for field in ["Σ", "J", "W", "V"]
        if !isnothing(myvcov_dict[field])
            myvcov_dict[field] = convert.(Float64, stack(myvcov_dict[field]))
        end
    end

    if isnothing(myvcov_dict["ses"])
        ses = nothing
    else
        ses = convert.(Float64, myvcov_dict["ses"])
    end

    myvcov = GMMvcov(
            method          = Symbol(myvcov_dict["method"]),
            V               = myvcov_dict["V"],
            ses             = ses,

            W               = myvcov_dict["W"],
            J               = myvcov_dict["J"],
            Σ               = myvcov_dict["Σ"])

    if myvcov.method == :bayesian_bootstrap

        # read CSV files
        weights_df = readdlm(full_path * "_boot_weights.csv",  ',', Float64)
        boot_fits_df = CSV.read(full_path * "_boot_fits_df.csv", DataFrame)
        boot_moms_hat_df = CSV.read(full_path * "_boot_moms_hat_df.csv", DataFrame)    
            
        all_boot_fits = [dict2fit(mybootfit_dict) for mybootfit_dict in myvcov_dict["boot_fits_dict"]]
        errored = [myfit.errored for myfit in all_boot_fits]
        n_errored = sum(errored)

        myvcov.boot_fits = GMMBootFits(
            errored = errored,
            n_errored = n_errored,
            boot_fits = all_boot_fits,
            boot_weights    = weights_df,
            boot_fits_df    = boot_fits_df,
            boot_moms_hat_df = boot_moms_hat_df)
    end

    @info "Read vcov type [" * string(myvcov.method) * "] from file from " * full_path 

    return myvcov
end