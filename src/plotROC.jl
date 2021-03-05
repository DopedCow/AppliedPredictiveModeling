using DataFrames
using DataFramesMeta
using MLBase
using VegaLite


#= ----------------------------------------------------------------------------

    plot_ROC_curve - plot ROC curve from vector of ROC points

---------------------------------------------------------------------------- =#
"""
    plot_ROC_curve(rc)

### Arguments

* `rc` : an `Array{ROCNums{Int64},1}`

### Returns

A VegaLite 

"""
function plot_ROC_curve(rc)

    df = DataFrame(tp_rate = Float64[],
                   fp_rate = Float64[])

    for r ∈ rc
        push!(df, (true_positive_rate(r), false_positive_rate(r)))
    end

    @orderby(df, :fp_rate, :tp_rate) |>
    @vlplot(
        height = 500,
        width = 500,
        title = {text = "ROC (Receiver Operating Characteristic) curve",
                 fontWeight = 400},
        mark = {:line, point = true, interpolate = "step-after"},
        x = {:fp_rate,
             title = "False positive rate (1 - specificity)"},
        y = {:tp_rate, title = "True positive rate (sensitivity)"},
        order = {field = [:fp_rate, :tp_rate]},
        config = {axis = {offset = 10},
                  font = "Cabin"}
    )
end

test = plot_ROC_curve(rc)

# Test functionality
gt = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0,
      0, 1, 1, 1, 0, 1, 0, 0, 1, 0]

scores = [0.7, 0.5, 0.5, 0.8, 0.6, 0.85, 0.3, 0.7, 0.4, 0.4,
          0.7, 0.5, 0.5, 0.8, 0.6, 0.85, 0.3, 0.35, 0.4, 0.4]

rc = roc(gt, scores, collect(0:0.05:1))
