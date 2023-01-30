


# """
# model function -- returns probability to choose route 1
# """
function logit_takeup_route1(
    theta::Vector{Float64};
    data::Matrix{Float64},
    model_params::Dict{String, Float64},
    MAXDIFF=200.0)

    alpha = theta[1]
    sigma = theta[2]
    price = model_params["price"] # this will be switched on only for treatment group

    travel_time_diff = data[:, 1]
    treated = data[:, 2]

    temp = ( - alpha .* travel_time_diff .+ price .* treated) ./ sigma
    temp = max.(min.(temp, MAXDIFF), -MAXDIFF)
    temp = exp.(temp)

    return temp ./ ( 1.0 .+ temp)
end

function data_synthetic(theta::Vector{Float64};
    data::Matrix{Float64},
    model_params::Dict{String, Float64}, asymptotic=false, rng=nothing)

    data_probs = logit_takeup_route1(theta; data=data, model_params=model_params)

    if asymptotic
        return data_probs
    end

    data_choices = rand(rng, size(data,1)) .<= data_probs
    data_choices = convert.(Float64, data_choices)

    return data_choices
end

# """
# Moments from model or data
# """
function moments(;
        takeup::Vector{Float64},
        data::Matrix{Float64})

    treated = data[:, 2]

    return hcat(takeup .* (1.0 .- treated),
                takeup .* treated)
end

# """
# moments (model minus data)
# """
function moments_gmm(;
    theta::Vector{Float64},
    mydata_dict::Dict{String, Array{Float64}},
    model_params::Dict{String, Float64})

    data = mydata_dict["data"]
    takeup_data = mydata_dict["takeup_data"]

    takeup_model = logit_takeup_route1(theta, data=data, model_params=model_params)

    moms_model = moments(takeup=takeup_model, data=data)
    moms_data  = moments(takeup=takeup_data, data=data)

    return moms_model .- moms_data
end


# Function to generate fake data for testing
function generate_data_logit(;N=200, travel_time_diff_max=20.0, price=10.0, rng=nothing)

    # travel time difference route 1 - route 0 (minutes)
    travel_time_diff = rand(rng, N) * travel_time_diff_max

    # define treatment group dummy (will get charges for route 0)
    treated = (rand(rng, N) .< 0.5).* 1.0

    # put together
    data = hcat(travel_time_diff, treated)

    # model parameters
    model_params = Dict(
        "price" => price
    )

    # generate data from model
    takeup_data = data_synthetic(true_theta, data=data, model_params=model_params, asymptotic=false, rng=rng)

    # this goes in GMM
    data_dict = Dict(
        "data" => data,
        "takeup_data" => takeup_data
    )

    return data_dict, model_params
end


## Define moment functions
function moms_data_cmd(data_dict)
    df = DataFrame(
    "takeup_data" => data_dict["takeup_data"],
    "treat" => data_dict["data"][:, 2],
    "control" => 1.0 .- data_dict["data"][:, 2]
    )
    # reg(df, term(:takeup_data) ~ term(:treat) + term(:control))

    mymodel = lm(@formula(takeup_data ~ 0 +  treat + control), df)
    
    M = coef(mymodel) |> transpose |> Matrix
    V = vcov(mymodel)

    return M, V
end

function moms_model_cmd(;
    mytheta::Vector{Float64},
    mydata_dict::Dict{String, Array{Float64}},
    model_params::Dict{String, Float64})

    data = mydata_dict["data"]
    # takeup_data = mydata_dict["takeup_data"]

    takeup_model = logit_takeup_route1(mytheta, data=data, model_params=model_params)

    # moms_model = moments(takeup=takeup_model, data=data)
    treat_vec = data[:, 2] .== 1.0
    contr_vec = data[:, 2] .== 0.0

    @assert all(treat_vec + contr_vec .== 1) 

    M = zeros(1, 2)
    M[1,1] = mean(takeup_model[treat_vec])
    M[1,2] = mean(takeup_model[contr_vec])

    return M
end