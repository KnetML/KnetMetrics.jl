module regression

using LinearAlgebra
using Statistics

include("metrics.jl"); export max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error

end
