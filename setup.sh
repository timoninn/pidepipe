#!/bin/sh

echo "Add $(pwd) to PYTHONPATH"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Current PYTHONPATH: $PYTHONPATH"
