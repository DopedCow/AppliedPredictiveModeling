"""
Julia implementation of the R code from
Applied Predictive Modeling
Chapter 6, Linear Regression and Its Cousins

by Morten Ørris Poulsen

Notes
For PCA look here: https://alan-turing-institute.github.io/DataScienceTutorials.jl/isl/lab-10/
For stats look here: https://juliastats.org

"""

# Introduction -------------------------------------------------------------
# This is an adaption – or update – of the code in section 6.5 of Applied
# Predictive Modeling. The aim is to use Julia for the code.

# Lessons Learned ----------------------------------------------------------
# - StatsBase requires manual handling of missing values

# - RData files can be read using RData package. Not currently implemented here.


# Setup --------------------------------------------------------------------
# using Colors
using CSV
using DataFrames
# using Gadfly
using GLM
using Random      #  provide seed(), …
using Statistics
using StatsBase
# using StatsModels

# Set random seed – 'LEGO' turned 180 degrees
Random.seed!(7390)


# Read data from TSV files -------------------------------------------------
sol_test_x = CSV.read("./data/solTestX.tsv", delim = "\t")
sol_test_x_trans = CSV.read("./data/solTestXtrans.tsv", delim = "\t")
sol_train_x = CSV.read("./data/solTrainX.tsv", delim = "\t")
sol_train_x_trans = CSV.read("./data/solTrainXtrans.tsv", delim = "\t")
sol_test_y = CSV.read("./data/solTestY.tsv", delim = "\t")
sol_train_y = CSV.read("./data/solTrainY.tsv", delim = "\t")


# Wrangle data -------------------------------------------------------------
sample(names(sol_train_x), 8)

train = hcat(sol_train_y, sol_train_x_trans)

describe(train)


# Train model --------------------------------------------------------------

# Build a string of all the predictors to form the RHS
recipe = Formula(:solTrainY, Expr(:call, :+, names(sol_train_x_trans)...))

model = GLM.lm(recipe, train)

GLM.r²(model)
GLM.adjr²(model)
GLM.dof(model)

lmpred = predict(model, sol_test_x_trans)

pv = collect(skipmissing(lmpred))

ov = collect(skipmissing(convert(Array, sol_test_y)))

# Calculate RMSD and R²
GLM.rmsd(pv, ov)
cor(pv, ov)^2


# Tests --------------------------------------------------------------------
