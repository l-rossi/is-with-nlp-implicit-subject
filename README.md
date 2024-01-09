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

Please use the root folder as your working directory.

## Goldstandard

https://docs.google.com/spreadsheets/d/1o92z018fu5IBF7pD2XGexKKM9XYjQSDHdDboLIzRwPQ/edit#gid=0

## Known bugs

### NominalizedVerbDetector

- evening could be a nominalized verb (to even) but is not in the context of 'In the evening, ...'

### All SpecializedInserters

- Capitalization for start of sentences is off if the sentence starts with a special character, i.e., a bullet for
  enumeration 

### DependentOfSameSentenceFilter
- wrongly filters "department" for process in example 8 of GS
