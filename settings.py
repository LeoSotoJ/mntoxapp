#!/usr/bin/env python
# coding: utf-8

# Select the MS dataset file

# MSrawdata = "xaf" #sample of the dataset
MSrawdata = "MassBank_NIST_Feb20" #entire dataset
# Tolerance for the cosine calculation with the CosineGreedy function matchms
cosine_tolerance=0.1 
# Molecular network parameters
# Similarity cut-off. The minimum similarity score for a pair of spectra to be connected within a network
cutoff=0.6
# Maximum number of edges per node. n-top cosine values.
links=10

