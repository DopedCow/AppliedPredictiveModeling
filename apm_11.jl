# --------------------------------------------------------------------------
# Julia implementation of the R code from
# Applied Predictive Modeling
# Chapter 11, Measuring Performance in Classification Models
#
# by Morten Ørris Poulsen
# --------------------------------------------------------------------------


# TODO Get original data file
# TODO Validate which packages are in actual use


# Introduction -------------------------------------------------------------
# This is an adaption – or update – of the code in section 11.4 of Applied
# Predictive Modeling. The aim is to use Julia for the code.


# Lessons Learned ----------------------------------------------------------
# -


# Setup --------------------------------------------------------------------

# using Colors
# using CSV
# using DataFrames
# using Gadfly
# using GLM
using Random      #  provide seed(), …
# using RData
# using Statistics
# using StatsBase
# using StatsModels

# Set random seed – 'LEGO' turned 180 degrees
Random.seed!(7390)


# Wrangle data -------------------------------------------------------------

# Create simuluated data: sim_train, sim_test


# Fit models ---------------------------------------------------------------


# Calculate sensitivity and specificity ------------------------------------


# Calculate confusion matrix -----------------------------------------------


# Create ROC curves --------------------------------------------------------


# Create lift charts -------------------------------------------------------


# Callibrate probabilities -------------------------------------------------
