# CAFO-AI
Code related to pilot study for detecting CAFOs in LMICs

These notebooks etc. are intended to illustrate the process used to obtain the results described in Mason, R., 2025, "Toward a global map of industrial pig and poultry farms: a pilot study".

Notebooks 1-8 illustrate the process of creating training and testing datasets. First, the locations of buildings/building clusters are determined, then Sentinel images are created.
  - Notebooks 1-5: Data exploration and creation of the training set of buildings in Iowa, Chile, Romania, and Mexico
  - Notebook 6: The held-out test regions
  - Notebook 7: The model application regions
  - Notebook 8: Sentinel data for all of the above.

Notebook 9: Model setup and training
Notebook 10: Correction of mislabeled training data based on the output of notebook 9 (then notebook 9 is rerun)
Notebook 11: Trained model is used to make predictions for the held-out test regions
Notebook 12: Trained model is used to make predictions for the unlabeled model application regions

utils.py, explore.py: Called by various notebooks.
sizeDistributions.py: Used to make figure 3 in the report

results_*.pkl are pickled geodataframes containing model CAFO probabilities and metadata for the four model application regions.

interactive_map_*.html are interactive maps for those regions, but they will only render for users with a GEE account whilst authenticated.
