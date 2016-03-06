# Base App

A very basic scaffold with scala and spark

## Get Started

In another tmux terminal, run these commands for automatic compiling
and execution: `~ run` The `Main.main` method is executed by default.

## HowTo:

DataSets are meant to be curated however needed and returned as
LabeledPoints for training

DataModels combine a data set and one of Spark's Machine Learning models
to create a customized learner. The class is overloaded at the moment,
but the intent is to streamline the process of creating a highly
accurate learner by providing an automated list of model parameters to
explore. Once a model has been optimized, it can be persisted to HDFS
(or some other store) and used for predicting unlabeled features.

In the future a module or companion project should be created to service
and API layer to the model wherein a web service can pass in unlabeled
features and return a prediction.
