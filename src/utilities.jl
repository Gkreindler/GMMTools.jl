### Initial conditions

"""
Creates a random matrix of initial conditions, taking bounds into account
"""
function random_initial_conditions(theta0::Vector, nic::Int; theta_lower=nothing, theta_upper=nothing)

    n_params = length(theta0)
    theta0_mat = zeros(nic, n_params)
    isnothing(theta_lower) && (theta_lower = fill(-Inf, n_params))
    isnothing(theta_upper) && (theta_upper = fill( Inf, n_params))

    for i=1:nic
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
theta_fix is a vector with the fixed values of the parameters, and missing at the other locations
this function fills in the missing values with values from theta_small
"""
function theta_add_fixed_values(theta_small, theta_fix)
    
    # theta_small elements are sometimes Dual or other types when doing AD
    theta = Vector{Number}(undef, length(theta_fix))

    theta[  ismissing.(theta_fix)] .= theta_small
    theta[.!ismissing.(theta_fix)] .= theta_fix[.!ismissing.(theta_fix)]

    return theta
end
