# cap-5602-final-project-group-5
Final project repository for FIU Fall 2022 group 5.
[Predict next-day rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

### Interesting tidbits:
To export a compatible environment that will work across all platforms:

`conda export --from-history > environment.yml`

To create an environment from the environment.yml:

`conda env update --prefix ./env --file environment.yml --prune`

To create a python runnable from a jupyter notebook to run on the command line:

`jupyter nbconvert --to python RainAus.ipynb`