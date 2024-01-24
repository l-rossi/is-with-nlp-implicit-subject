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

## Known bugs

### NominalizedVerbDetector

- evening could be a nominalized verb (to even) but is not in the context of 'In the evening, ...'

### All SpecializedInserters

- Capitalization for start of sentences is off if the sentence starts with a special character, i.e., a bullet for
  enumeration 

### DependentOfSameSentenceFilter
- wrongly filters "department" for process in example 8 of GS

## Some Stuff to be deleted later
TODO:
-	Morphology for 
-	If gerund is separated by comma from its head, it seems like no subject is allowed


45 for example of insertion closer to predicate



Example 24 bugged casing
Example 24 gerund not needing 


35 imperative too greedy

Perplexity filter too strict, for example: codling moth pruned in 18. Assuming prior probability of generating "codling moth" is just very low (perplexity ration 427/159 !). ChatGPT 3.5 seems to be able to handle this though

GERUNDS:
6, 7, 10, 11, 21, 24, 30, 32, 43, 46, 47, 49
