#= -------------------------------------------------------------------------
    Julia Implementation of the R code in Applied Predictive Modeling
    Chapter 5, Measuring Performance in Regression Models
    by Morten Ørris Poulsen

    This is an adaption – or update – of the code in section 5.3 of Applied
    Predictive Modeling. The aim is to use Julia for the code.
------------------------------------------------------------------------- =#


#= Lessons Learned ---------------------------------------------------------
    - The min() and max() functions should be replaced by minimum() and
    maximum() as the former do not work as expected on arrays.
    - The is an r²() function i StatsBase, but it takes an
    obj::StatisticalModel as input. Have not yet succeeded in creating this,
    so for now it is calculated using the square of the correlation.
------------------------------------------------------------------------- =#


# Setup --------------------------------------------------------------------
using Colors
using DataFrames
using Gadfly
using Random      #  provide seed(), …
using Statistics
using StatsBase
using StatsModels

# Set random seed – 'LEGO' turned 180 degrees
Random.seed!(7390)


# Wrangle Data -------------------------------------------------------------

# Create vectors: observed, predicted
observed = [0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.40, 0.62, 0.99,
            -0.18, 0.32, 0.34, -0.30, 0.04, -0.87, 0.55, -1.30, -1.15, 0.20]

predicted = [0.24, 0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43, 0.49, 0.79,
             -1.19, 0.06, 0.75, -0.07, 0.43, -0.42, -0.25, -0.64, -1.26,
             -0.07]

# Calculate residual values: residuals
residuals = observed - predicted

all_numbers = vcat(observed, predicted)

maximum(all_numbers)
minimum(all_numbers)

StatsBase.summarystats(residuals)
StatsBase.describe(residuals)

#= Consider extending the range artificially similar to extendedrange() in R
https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/extendrange
=#

# This version is a copy of the R function but not done
function extend_range(x::Array, r = maximum(x) - minimum(x), f::Float64 = 0.05)::Array
    extension = r * f
    start = r[1] - extension
    finish = r[2] + extension
    return [start, finish]
end

# This is a simpler version and works
function extend_range(x::Array, f::Float64 = 0.05)::Array
    highest = maximum(x)
    lowest = minimum(x)    
    range = highest - lowest
    extension = range * f
    start = lowest - extension
    finish = highest + extension
    return [start, finish]
end
    
new_range = extend_range(all_numbers)
extend_range([1, 7, 5])

# Plot using Gadfly with a data frame
data = DataFrames.DataFrame(obs = observed, pred = predicted)
Gadfly.plot(data, x = :obs, y = :pred, Geom.point)

# The same but with a prettier range
Gadfly.plot(data, x = :obs, y = :pred, 
            Coord.cartesian(xmin=new_range[1], xmax=new_range[2], ymin=new_range[1], ymax=new_range[2]),
            Geom.point)

# The same using the arrays directly
Gadfly.plot(x = observed, y = predicted, color = [HSL(0, 0, 0.7)], Geom.point,
            intercept = [0], slope = [1], Geom.abline(color = HSL(0, 0, 0.7),
                                                      style = :dot))

Gadfly.plot(x = predicted, y = residuals, color = [HSL(0, 0, 0.7)], Geom.point,
            intercept = [0], slope = [0], Geom.abline(color = HSL(0, 0, 0.7),
                                                      style = :dot))


# Calculate performance ----------------------------------------------------

# Calculate R squared (the correlation squared)
Statistics.cor(predicted, observed)^2

# Calculate Root Mean Squared Deviation (same as RMSE)
StatsBase.rmsd(predicted, observed)
#RMSE(predicted, observed)

# Calculate simple correlation. Square to get R2
Statistics.cor(predicted, observed)

# Calculate rank correlation - aka. Spearman
StatsBase.corspearman(predicted, observed)
