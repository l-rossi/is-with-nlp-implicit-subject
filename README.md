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

You will need an OpenAI API key environment variable (`OPENAI_API_KEY`) to use the ChatGPTFilter. You can provided it in a `.env` file:

```
OPENAI_API_KEY=<Your Key>
```

## Usage
The main entry point to this code base is the `ImplicitSubjectPipeline'. A minimal example of its usage
can be seen in the following:

```python
pipeline = ImplicitSubjectPipeline(
    missing_subject_detectors=[...],
    candidate_extractor=CandidateExtractorImpl(),
    candidate_rankers=[...],
    missing_subject_inserter=ImplicitSubjectInserterImpl(),
)

result = pipeline.apply("<The inspected text>", "<The context from which to extract subjects>")
```




## Goldstandard

https://docs.google.com/spreadsheets/d/1o92z018fu5IBF7pD2XGexKKM9XYjQSDHdDboLIzRwPQ/edit#gid=0
