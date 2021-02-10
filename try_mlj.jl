# Load a selection of features and labels from the Ames House Price dataset:

using MLJ
X, y = @load_reduced_ames
booster = @load EvoTreeRegressor

# Change default parameters
booster.max_depth = 2
booster.nrounds=50

# Combine the model with categorical feature encoding:
pipe = @pipeline ContinuousEncoder booster

# Define a hyper-parameter range for optimization:
max_depth_range = range(pipe,
                        :(evo_tree_regressor.max_depth),
                        lower = 1,
                        upper = 10)

# Wrap the pipeline model in an optimization strategy:
self_tuning_pipe = TunedModel(model=pipe,
                              tuning=RandomSearch(),
                              ranges = max_depth_range,
                              resampling=CV(nfolds=3, rng=456),
                              measure=l1,
                              acceleration=CPUThreads(),
                              n=50)

# Bind the "self-tuning" pipeline model (just a container for hyper-parameters) to data in a machine (which will additionally store learned parameters):
mach = machine(self_tuning_pipe, X, y)

# Evaluate the "self-tuning" pipeline model's performance (implies nested resampling):

evaluate!(mach,
          measures=[l1, l2],
          resampling=CV(nfolds=6, rng=123),
          acceleration=CPUProcesses(), verbosity=2)