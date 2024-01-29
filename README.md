Approaching Information System Challenges with Natural Language Processing - Finding Implicit Subjects
==============================

This project tackles the challenge of identifying implicit subjects in legal text by employing a rule based approach

## Installation

I recommend using [Anaconda](https://www.anaconda.com/) for setting up the environment and all dependencies (I had some slight problems with the solver in conda version 23.9.0 on Ubuntu. I solved them by using libmamba).

After cloning the project, run from inside the cloned directory:

```console
conda env create -f env.yml
```

to create the `is-with-nlp-implicit-subject` environment on your machine and:

```console
conda activate is-with-nlp-implicit-subject
```

to select it.


If this fails, you can fall back to the `requirements.txt` and a Python 3.10
installation.hmm


I would also recommend removing the `is-with-nlp-implicit-subject` when you
are done with this project as it uses up a substantial chunk of disk space.

## Running

Please use the root folder as your working directory. When executing the script, for example:

```console
python ./src/main.py
```

You will need an OpenAI API key environment variable (`OPENAI_API_KEY`) to use the `ChatGPTFilter`. You can provide it
in
a `.env` file (An entire pass through the gold standard costs approximately 6ct):

```
OPENAI_API_KEY=<Your Key>
```

Please note, that running the program for the first time will install some extra files and may thus take longer.

## Usage

The main entry point to this code base is the `ImplicitSubjectPipeline'. A minimal example of its usage can be seen in
the following:

```python
pipeline = ImplicitSubjectPipeline(
    missing_subject_detectors=[...],
    candidate_rankers=[...],
)

result = pipeline.apply("<The inspected text>", "<The context from which to extract subjects>")
```

| Parameter                   | Interface                 | Implementations                                                                                                                                                                                                            |
|-----------------------------|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `missing_subject_detectors` | `ImplicitSubjectDetector` | `PassiveDetector`, `ImperativeDetector`, `GerundDetector`, `NominalizedGerundWordlistDetector`, `NounVerbStemDetector`                                                                                                     |
| `candidate_rankers`         | `CandidateFilter`         | `ImperativeFilter`, `PartOfSpeechFilter`, `DependentOfSameSentenceFilter`, `ChatGPTFilter`,`SimilarityFilter`, `PerplexityFilter`, `PreviouslyMentionedRelationFilter`, `CandidateTextOccurrenceFilter`, `ProximityFilter` |

For more information on the configuration, please refer to the doc strings.
For more information on the components refer either to the doc strings or preferrably to the writeup
at `documentation/writeup/is_with_nlp_is.pdf`.

An example usage of the pipeline can be found in `main.py` which runs an implicit subject pipeline against our gold
standard.

## Goldstandard

The gold standard can be found in `./data/evaluation/gold_standard.csv`:

| Column           | Meaning                                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------------------|
| Source           | The file from to use as context for the pipeline (also the place from which the inspected sentence is taken). |
| Input Data       | The sentence or text fragment to inspect.                                                                     |
| Gold Standard    | The expected output of the pipeline, i.e., the inspected sentence with all implicit subjects made explicit.   |
| Implicit Subject | A list of the correct subjects for insertion                                                                  |
| Target           | A list of the targets, i.e., the part of the sentence that requires the insertion of a subject.               |

The sources referenced in the gold standard are relative to the `data/external` directory.

## Index of important files

| File                        | Location                                             |
|-----------------------------|------------------------------------------------------|
| Paper                       | `./documentation/writeup/is_with_nlp_is.pdf`         |
| Final Presentation          | `./documentation/final_presentation/IS WITH NLP.pdf` |
| Final Presentation (Online) | [Pitch.com](https://pitch.com/v/is-with-nlp-58ufnj)  |
| Gold Standard               | `./data/evaluation/gold_standard.csv`                |
