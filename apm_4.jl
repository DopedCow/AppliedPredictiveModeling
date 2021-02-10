"""

# Details

Julia implementation of the R code in Applied Predictive Modeling
Chapter 4 – Over-Fitting and Model Tuning

"""

# Setup --------------------------------------------------------------------
using DataFrames
using LIBSVM
using MLJ
using MLJBase
using NearestNeighborModels
using Random
using RData
using Statistics

using DataTables
using Query
using VegaLite
using VegaDatasets



"""
4.9 Computing
"""
Random.seed!(7390)

# Setup the model
knn = @load KNNClassifier

# Load the dataset (Dict with a dataframe and an array)
twoClass = RData.load("./data/twoClassData.RData")
predictors = twoClass["predictors"]
classes = twoClass["classes"]

# Split into train and test
train, test = partition(eachindex(classes), 0.8, shuffle = true)
trainPredictors = predictors[train, :]
trainClasses = classes[train, :]
testPredictors = predictors[test, :]
testClasses = classes[test, :]

describe(trainPredictors)
describe(testPredictors)

X, y = @load_boston
y
knn = @load KNNRegressor

evaluate(knn, X, y, resampling=CV(nfolds=5), measure=[rms, mae])

mach = machine(knn, X, y)

evaluate(knn, predictors, classes, resampling=CV(nfolds=5), 
         measure=[rms, mae], check_measure = false)






