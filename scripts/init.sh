#!/bin/sh


echo "Initiate project"


# Data
mkdir -p data/{raw,interim/{train,dev,test},processed/{train,dev,test},submissions,external}


# Docs
mkdir -p docs

touch docs/.gitkeep


# Source
mkdir -p src/{data,features,models,visualization}

touch src/__init__.py

touch src/data/make_dataset.py

touch src/features/build_features.py

touch src/models/train.py
touch src/models/predict.py


# Models
mkdir -p models

touch models/.gitkeep


# Notebooks
mkdir -p notebooks/{exploratory,reports}

touch notebooks/{exploratory,reports}/.gitkeep


# References
mkdir -p references

touch references/.gitkeep


# Reports
mkdir -p reports/figures

touch reports/figures/.gitkeep


# Make
touch Makefile