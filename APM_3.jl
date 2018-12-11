# --------------------------------------------------------------------------
# Julia implementation of the R code in Applied Predictive Modeling
# Chapter 3 – Data Pre-processing
# Code from chapter 3.8
# --------------------------------------------------------------------------


# Setup --------------------------------------------------------------------
using DataFrames
using DataFramesMeta
using RData
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


# Transform data -----------------------------------------------------------
StatsBase.skewness(segdata[:AngleCh1])


# Filter data --------------------------------------------------------------


# Create dummy variables ---------------------------------------------------


# Exercise 3.1 -------------------------------------------------------------


# Exercise 3.2 -------------------------------------------------------------


# Exercise 3.3 -------------------------------------------------------------
