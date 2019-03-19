# --------------------------------------------------------------------------
# Julia implementation of the R code in Applied Predictive Modeling
# Chapter 3 – Data Pre-processing
# Code from chapter 3.8

# --------------------------------------------------------------------------


# Setup --------------------------------------------------------------------
using DataFrames
using DataFramesMeta
using RData
using Statistics
using StatsBase


# Get data -----------------------------------------------------------------
rdata = RData.load("./data/segmentationOriginal.RData")
df = rdata["segmentationOriginal"]

segdata = @linq df |>
    where(:Case .== "Train")

cell = segdata[:Cell]
class = segdata[:Class]
case = segdata[:Case]

# TODO Remove columns
segdata = segdata[:, 4:end]

# With Julia 1.0, contains() is replaced by occursin() for strings
# Note the dot to operate on the items individually
status_columns = occursin.("Status", string.(names(segdata)))


status_columns .== true

countmap(status_columns)

# Transform data -----------------------------------------------------------
StatsBase.skewness(segdata[:AngleCh1])

aggregate(segdata, skewness)

aggregate(segdata, :Cell, skewness)

head(segdata[[:AreaCh1]])


# Filter data --------------------------------------------------------------


# Create dummy variables ---------------------------------------------------


# Exercise 3.1 -------------------------------------------------------------


# Exercise 3.2 -------------------------------------------------------------


# Exercise 3.3 -------------------------------------------------------------
