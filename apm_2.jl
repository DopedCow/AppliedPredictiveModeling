"""

# Details

Julia implementation of the R code in Applied Predictive Modeling
Chapter 2 – Data Pre-processing

There is no code in chapter 2, so this is an attempt at recreating similar
plots as those that appear througout the chapter.

When VegaLite documentation suggests using Null, in Julia it becomes nothing.

https://vega.github.io/vega-lite/docs/format.html

Have collected the original data from:
https://fueleconomy.gov/feg/epadata/vehicles.csv
"""

# Setup --------------------------------------------------------------------
using CSVFiles
using DataTables
using GLM
using HTTP
using Query
using VegaLite
using VegaDatasets



"""
Code for plot 2.1
"""
cars = VegaDatasets.dataset("cars")

plot_02_01 = cars |>
    @filter(_.Year < "1972-01-01") |>
    @vlplot(
        mark = :point,
        column = :Year,
        x=:Displacement,
        y=:Miles_per_Gallon,
        width=400,
        height=400
)

plot_02_01 |> save("figure_02_01.svg")


"""
Code for plot 2.2a
"""
plot_02_02 = cars |>
    @vlplot(
        width=400,
        height=400,
        layer = [
            {
            :point,
            x=:Displacement,
            y=:Miles_per_Gallon
            },

            {
            transform=[
                {
                    regression = :Miles_per_Gallon,
                    on = :Displacement
                }
            ],
            mark={:line, color="firebrick"},
            x="Displacement:q",
            y="Miles_per_Gallon:q"
            },
])

plot_02_02 |> save("figure_02_02.svg")


"""
Code for plot 2.2b
"""
training = cars |> @filter(_.Year <= "1971-01-01")
testing = cars |> @filter("1970-01-01" < _.Year < "1972-01-01")

linearRegressor = lm(@formula(Miles_per_Gallon ~ Displacement), training)
linearFit = predict(linearRegressor)
test = DataTable(training)

cars_dt = DataTable(observed = test.Miles_per_Gallon,
                    predicted = linearFit)


cars_dt |>
    @vlplot(
        mark = :point,
        x = :observed,
        y = :predicted
)



# Plot 2.2b ----------------------------------------------------------------
"""
New code using the lates download of the data.
"""
vehicles = DataTable(CSVFiles.load("./data/vehicles.csv"))

test = vehicles |>
    @select(:cylinders, :displ, :drive, :fuelType, :highway08, :make, :model,
            :year)
