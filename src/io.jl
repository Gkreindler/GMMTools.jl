


function Base.show(io::IO, r::GMMFit)
    println("GMMResult object with fields: thata0, W, weights, N, all_model_fits, vcov, fit_step1, etc.")
    println("  theta_names:", r.theta_names)
    println("  thata_hat:  ", r.theta_hat)
    println("Optimization results:")
    println("  converged:  ", r.converged)
    println("  obj_value:  ", r.obj_value)
    println("  iterations:  ", r.iterations)
    println("  iteration_limit_reached:  ", r.iteration_limit_reached)
end

function Base.show(io::IO, r::GMMBootFits)
    println("Baysian bootstrap results struct with the following fields:")
    for f in fieldnames(GMMBootFits)
        # println("  ", f, ": ", getfield(r, ))
        println("...field ", f, " is a ", typeof(getfield(r, f)))
    end
    # display(r.all_model_fits)
    # display(r.all_boot_fits)
end

### Writing and Reading GMMFit objects

to_dict(myfit::Union{GMMFit, GMMvcov}) = Dict(k => getfield(myfit, k) for k ∈ fieldnames(typeof(myfit)))
to_dict(boot_fits::Vector{GMMFit}) = [to_dict(boot_fits[i]) for i=1:length(boot_fits)]


function parse_json(myfit::Union{GMMFit, GMMvcov})

    # convert to dictionary
    myfit_dict = to_dict(myfit)

    # write as JSON
    return JSON.json(myfit_dict, 2)
end



"""
Write GMMFit object to files: JSON for most fields + CSV for all_model_fits table
All paths should exist.
"""
function write(myfit::GMMFit, opts::GMMOptions; subpath="fit")
    
    # paths
    if opts.path == "" 
        return
    end
    full_path = opts.path
    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath

    myfit = deepcopy(myfit)

    # write vcov, then remove
        if !isnothing(myfit.vcov)
            write(myfit.vcov, opts)
            myfit.vcov = nothing
        end
    
    # write table(s), then remove
        if isnothing(myfit.all_model_fits)
            myfit.all_model_fits = table(myfit)
        end
        CSV.write(full_path * ".csv", myfit.all_model_fits)
        myfit.all_model_fits = nothing

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

function write(myvcov::GMMvcov, opts::GMMOptions; subpath="vcov")
    
    if opts.path == ""
        return
    end
    full_path = opts.path
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
end






### Reading GMMFit objects

function parse_vector(s::AbstractString)
    return parse.(Float64, split(s[2:(end-1)],","))
end

function parse_weight_matrix(W)
    if isa(W, Dict) && ("λ" ∈ keys(W))
        return I
    else

        # JSON writes the matrix as a vector of vectors
        return convert.(Float64, stack(W))
    end
end

function dict2fit(myfit_dict)

    # JSON.print(myfit_dict, 2)

    if isnothing(myfit_dict["weights"])
        weights = nothing
    else
        weights = convert.(Float64, myfit_dict["weights"])
    end

    myfit = GMMFit(
            theta0          =convert.(Float64, myfit_dict["theta0"]),
            theta_hat       =convert.(Float64, myfit_dict["theta_hat"]),
            theta_names     =myfit_dict["theta_names"],

            moms_hat        = myfit_dict["moms_hat"],
            n_obs           =myfit_dict["n_obs"],
            n_moms          =myfit_dict["n_moms"],

            mode            =Symbol(myfit_dict["mode"]),
            weights         =weights,
            W               =parse_weight_matrix(myfit_dict["W"]),

            obj_value       =convert.(Float64, myfit_dict["obj_value"]),
            converged       =myfit_dict["converged"],
            iterations      =myfit_dict["iterations"],
            iteration_limit_reached=myfit_dict["iteration_limit_reached"],
            time_it_took    =myfit_dict["time_it_took"],

            idx             =myfit_dict["idx"])

    return myfit
end

"""
filepath should not include the extension (.csv or .json)
"""
function read_fit(opts::GMMOptions; subpath="fit")

    full_path = opts.path
    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath
    
    # files exist?
    files_exist = isfile(full_path * ".csv") && isfile(full_path * ".json")

    if !files_exist
        println("files `fit.csv` and/or `fit.json` do not exist in ", full_path)
        return nothing
    end

    # read and parse JSON file
    myfit_dict = JSON.parsefile(full_path * ".json")
    myfit = dict2fit(myfit_dict)

    # read CSV file with estimates
    df = CSV.read(full_path * ".csv", DataFrame)
    myfit.all_model_fits = df

    # read CSV file with moments at theta_hat
    if isfile(full_path * "_moms_hat.csv")
        myfit.moms_hat = readdlm(full_path * "_moms_hat.csv", ',', Float64)
    end

    return myfit
end


function read_vcov(opts::GMMOptions; subpath="vcov")
    
    full_path = opts.path
    (full_path[end] == '/') && (full_path *= '/') # ? platform issues?
    full_path *= subpath

    # files exits?
    if !isfile(full_path * ".json")
        println("file `vcov.json` does not exist in ", full_path)
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
                
        myvcov.boot_fits = GMMBootFits(
            boot_fits = [dict2fit(mybootfit_dict) for mybootfit_dict in myvcov_dict["boot_fits_dict"]],
            boot_weights    = weights_df,
            boot_fits_df    = boot_fits_df,
            boot_moms_hat_df = boot_moms_hat_df)
    end

    return myvcov
end