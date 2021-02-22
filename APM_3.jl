"""
Julia implementation of the R code in Applied Predictive Modeling
Chapter 3 – Data Pre-processing
Code from chapter 3.8

# Notes

"""

# Setup --------------------------------------------------------------------
using BoxCoxTrans
using DataFrames
using DataFramesMeta
using RData
using StatsBase

include("./src/APL.jl")


# Get data -----------------------------------------------------------------
df = APLData("Segmentation")


segdata = @linq df |>
    where(:Case .== "Train")

cell = segdata[:Cell]
class = segdata[:Class]
case = segdata[:Case]

# Remove columns 1 to 3
segdata = segdata[:, 4:end]

# Remove the status columns
# Note the dot to operate on the items individually
status_columns = occursin.("Status", string.(names(segdata)))

# countmap(status_columns)
select!(segdata, Not(names(segdata)[status_columns]))


#= Transform data =#

# Calculate skewness
StatsBase.skewness(segdata[:AngleCh1])

aggregate(segdata, skewness)

Ch1AreaTrans = BoxCoxTrans.transform(segdata.AreaCh1)
BoxCoxTrans.lambda(segdata.AreaCh1).value


#aggregate(segdata, :Cell, skewness)

#head(segdata[[:AreaCh1]])


# Filter data --------------------------------------------------------------


# Create dummy variables ---------------------------------------------------


# Exercise 3.1 -------------------------------------------------------------


# Exercise 3.2 -------------------------------------------------------------


# Exercise 3.3 -------------------------------------------------------------



using Plots
plotly() # using plotly for 3D-interacive graphing


# split half to training set
Xtr = convert(Array, iris[1:2:end, 1:4])
Xtr_labels = convert(Array, iris[1:2:end, 5])

# split other half to testing set
Xte = convert(Array, iris[2:2:end, 1:4])
Xte_labels = convert(Array, iris[2:2:end, 5])



# apply PCA model to testing set
Yte = MultivariateStats.transform(M, Xte)

# reconstruct testing observations (approximately)
Xr = reconstruct(M, Yte)

# group results by testing set labels for color coding
setosa = Yte[:, Xte_labels.=="setosa"]
versicolor = Yte[:, Xte_labels.=="versicolor"]
virginica = Yte[:, Xte_labels.=="virginica"]

# visualize first 3 principal components in 3D interacive plot
p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)
plot!(p, xlabel="PC1", ylabel="PC2", zlabel="PC3")



#=
PCA using MLJ.jl
=#
using MLJ
using MLJMultivariateStatsInterface

X, y = @load_iris
#schema(X)

train, test = partition(eachindex(y), 0.50, shuffle=true, rng = 7390)
# ([125, 100, 130, 9, 70, 148, 39, 64, 6, 107  …  110, 59, 139, 21, 112, 144, 140, 72, 109, 41], [106, 147, 47, 5])
# Instantiate and fit the model/machine:

# prøve = @load PCA
pca = PCA(maxoutdim = 3)
mach = machine(pca, X)
fit!(mach, rows = train)

#Transform selected data bound to the machine:
y_test = transform(mach, rows=test)

#Transform new data:
Xnew = (sepal_length=rand(3), sepal_width=rand(3),
        petal_length=rand(3), petal_width=rand(3))

transform(mach, Xnew)
#(x1 = [4.316489556017327, 4.238914682299775, 5.342670097730145],
# x2 = [-5.075190145248834, -5.225553878737216, -5.318389797856577],)