Approaching Information System Challenges with Natural Language Processing - Finding Implicit Subjects
==============================

This project tackles the challenge of identifying implicit subjects in legal text by employing a rule based approach

## Installation

I recommend using [Anaconda](https://www.anaconda.com/) for setting up the environment and all dependencies.

Run:

```console
conda env create -f env.yml
```

to create the `is-with-nlp-implicit-subject` environment on your machine and:

```console
conda activate is-with-nlp-implicit-subject
```

to select it.

coreferee requires an additional download which can be done from the console (with an active Anaconda environment.
spaCys packages should be installed by conda):

```console
python -m coreferee install en
```

## Project Plan

Two major groups of implicit subjects identified so far:
a) endophoric references b) Omitted "agent" in passive sentences

There exits a plethora of coreference corpora but as far as I can see none a
specific to the domain of legal texts. But too specific for our use case imo-

I want to have the Insurance Claims regulatory
paragraphs; process
descriptions data


Coreferee has a great discussion of how rules are used to resolve coreferences: https://github.com/msg-systems/coreferee. (Which then uses an ML at the end)

-> Question then is how to integrate the noun phrases into the text.

Q1: Does Grammar actually matter here? At least early approaches seem to be fine with lemmata in input.
Q2: Deal with endophoric phrases: ``he or she`` creates problems as only one of the pronouns is resolved.

According to [Ji at](https://www.sciencedirect.com/science/article/pii/S0306457320308608) al 4 types of resolution (Their stuff is for implicit subject in dialog):
1) mention-pair models (connect nodes of graph over pairwise probabilities)
2) entity-level models -> augment previous model by merging based on all the edges between the clusters?
3) latent-tree models -> ??? Clustering using SVM
4) mention-ranking models -> Same just with all tokens??




### Just some notes




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
