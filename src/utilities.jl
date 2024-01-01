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

            if (θL == -Inf) & (θH == Inf) # without bounds, use the initial value * 1.5. If the initial value is zero, default to uniform over [-1.5, 1.5]
                theta0_mat[i,j] = (-1.5 + 3.0 * rand()) * (theta0[j] + (theta0[j] ≈ 0.0))

            elseif (θL > -Inf) & (θH == Inf) # lower bound + 50-150% of (initial value minus lower bound)
                theta0_mat[i,j] = θL + (theta0[j] - θL) * (0.5 + rand())

            elseif (θL == -Inf) & (θH < Inf)
                theta0_mat[i,j] = θH - (θH - theta0[j]) * (0.5 + rand())

            else # uniform between lower and upper bound
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

Note: all values of theta_large are of the same type as theta_small[1]. This typically works with AD, where all elements of theta_small are Dual numbers. (But theta_fix will contain Float64s)
"""
function theta_add_fixed_values(theta_small, theta_fix)
    
    # theta_small elements are sometimes Dual or other types when doing AD
    theta_large = Vector{typeof(theta_small[1])}(undef, length(theta_fix))
    # theta = Vector{Number}(undef, length(theta_fix))

    theta_fix_idxs = .!ismissing.(theta_fix)
    theta_large[.!theta_fix_idxs] .= theta_small
    theta_large[  theta_fix_idxs] .= theta_fix[theta_fix_idxs]

    return theta_large
end

default_theta_names(n_params) = ["theta_$i" for i=1:n_params]