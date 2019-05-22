#!/bin/sh


echo "Initiate project tree"

mkdir -p data/{raw,interim/{train,dev,test},processed/{train,dev,test},submissions}
mkdir -p srs/{data,features,models,visualization}
mkdir -p models
mkdir -p notebooks/{exploratory,reports}
mkdir -p references
mkdir -p reports/figures
