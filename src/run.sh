#!/bin/bash

# Loop over a list of configuration files.
for config in ../configs/*.json; do
    echo "Running experiment with configuration: $config"
    python3 train.py --config "$config"
done