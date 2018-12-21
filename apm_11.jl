# --------------------------------------------------------------------------
# Julia implementation of the R code from
# Applied Predictive Modeling
# Chapter 11, Measuring Performance in Classification Models
#
# by Morten Ørris Poulsen
# --------------------------------------------------------------------------


# TODO Validate which packages are in actual use
# TODO Find package that has the QDA function. DiscriminantAnalysis fails


# Introduction -------------------------------------------------------------
# This is an adaption – or update – of the code in section 11.4 of Applied
# Predictive Modeling. The aim is to use Julia for the code.


# Lessons Learned ----------------------------------------------------------
#   - The . is extremely important in Julia, when you are used to working
#     with R.


# Setup --------------------------------------------------------------------

# using Colors
using DataFrames
using DataFramesMeta
using DecisionTree
using Distributions
# using Gadfly
using MLBase
using Random
using StatsBase

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

features = convert(Array, sim_train[:, 1:2])
labels = convert(Array, sim_train[:, 4])

# model = DecisionTreeClassifier()
model = build_forest(labels, features, 2, 2000, 0.5, 5)

# apply learned model
apply_forest(model, [-0.23, -0.43])
apply_forest(model, features)

# get the probability of each label
apply_forest_proba(model, [-0.23, -0.43], ["Class1", "Class2"])
apply_forest_proba(model, features, ["Class1", "Class2"])

n_folds=3; n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

# Calculate sensitivity and specificity ------------------------------------


# Calculate confusion matrix -----------------------------------------------


# Create ROC curves --------------------------------------------------------


# Create lift charts -------------------------------------------------------


# Callibrate probabilities -------------------------------------------------
