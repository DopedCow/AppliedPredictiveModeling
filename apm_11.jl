# --------------------------------------------------------------------------
# Julia implementation of the R code from
# Applied Predictive Modeling
# Chapter 11, Measuring Performance in Classification Models
#
# by Morten Ørris Poulsen
# --------------------------------------------------------------------------


# TODO Get original data file, if any
# TODO Validate which packages are in actual use


# Introduction -------------------------------------------------------------
# This is an adaption – or update – of the code in section 11.4 of Applied
# Predictive Modeling. The aim is to use Julia for the code.


# Lessons Learned ----------------------------------------------------------
# -


# Setup --------------------------------------------------------------------

# using Colors
using DataFrames
using DataFramesMeta
using DecisionTree
using Distributions
# using Gadfly
using MLBase
using Random      #  provide seed(), …
# using Statistics
# using StatsBase
# using StatsModels

# Set random seed – 'LEGO' turned 180 degrees
Random.seed!(7390)


# Wrangle data -------------------------------------------------------------

# quad_boundary() is a re-implementation of quadBoundaryFunc from the R package
# AppliedPredictiveModeling

function quad_boundary(n)
  sigma = [1 0.7; 0.7 2]

  tmpdata = DataFrames.DataFrame(transpose(rand(MvNormal(sigma), n)))

  zfoo(x, y) = - 1 - 2 * x - 0 * y - .2 * x ^ 2 + 2 * y ^ 2

  z2p(x) = 1 / (1 + exp(-x))

  prob = z2p.(zfoo.(tmpdata[:x1], tmpdata[:x2]))
  class = categorical(ifelse.(rand(n) .<= prob, "Class1", "Class2"))

  @transform(tmpdata, prob = prob, class = class)
end

# Create simuluated data: sim_train, sim_test
sim_train = quad_boundary(500)
sim_test = quad_boundary(1000)

head(sim_train)


# Fit models ---------------------------------------------------------------

# https://github.com/bensadeghi/DecisionTree.jl
rfmodel = RandomForestClassifier(nsubfeatures = 3, ntrees = 2000, partialsampling=0.7, maxdepth = 4)
rfmodel = build_forest(convert(Array, sim_train[:class]), sim_train[:x1], 20, 50, 1.0)

rfmodel = RandomForestClassifier(n_estimators = 3, max_depth = 4)

# Calculate sensitivity and specificity ------------------------------------


# Calculate confusion matrix -----------------------------------------------


# Create ROC curves --------------------------------------------------------


# Create lift charts -------------------------------------------------------


# Callibrate probabilities -------------------------------------------------
