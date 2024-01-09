import spacy

from ImplicitSubjectPipeline import ImplicitSubjectPipeline
from candidate_extraction.CandidateExtracorImpl import CandidateExtractorImpl
from candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from candidate_filtering.ImperativeFilter import ImperativeFilter
from candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from candidate_filtering.PerplexityFilter import PerplexityFilter
from insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from missing_subject_detection.GerundDetector import GerundDetector
from missing_subject_detection.ImperativeDetector import ImperativeDetector
from missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from missing_subject_detection.PassiveDetector import PassiveDetector
from util import load_gold_standard


def main():
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[PassiveDetector(), ImperativeDetector(), GerundDetector(),
                                   NominalizedGerundWordlistDetector()],
        candidate_extractor=CandidateExtractorImpl(),
        candidate_rankers=[
            ImperativeFilter(),
            PartOfSpeechFilter(),
            DependentOfSameSentenceFilter(),
            PerplexityFilter(max_returned=10000),
            # SimilarityFilter(),
            CandidateTextOccurrenceFilter(),
            # , SemanticSimilarityFilter(), SubjectIfSameSentenceFilter(),
            # ProximityFilter()
        ],
        missing_subject_inserter=ImplicitSubjectInserterImpl(),
        verbose=True
    )

    similarity_nlp = spacy.load("en_core_web_lg")
    n_inspected = 0
    n_correct = 0

    mask = ""
    for i, (source, target, gs) in enumerate(list(load_gold_standard())[24:25]):
        print(f"Enter {i}")
        print("Context:")
        print(source)
        print("-" * 5)
        print("Inspected text:")
        print(target)
        print("-" * 5)

        generated = pipeline.apply(
            inspected_text=target,
            context=source
        )

        similarity = similarity_nlp(gs).similarity(similarity_nlp(generated))
        n_inspected += 1
        if gs.strip() == generated.strip():
            n_correct += 1
            mask += "x"
        elif similarity > 0.995:  # <- this is just some magic number based on empirical observation. This is only an indication of possbile easy improvements and not really a metric for the process doing a good job as sentence length seems to skew this metric.
            mask += "-"
        else:
            mask += "_"

        print("-" * 4)
        print("Expected:", gs)
        print("Actual:  ", generated)
        print("Similarity: ", similarity)

        print("-" * 9)

    print(mask)
    result_txt = f"Correct: {n_correct}/{n_inspected} ({n_correct / n_inspected * 100 :.2f}%)"
    print(result_txt)
    with open("./log/res", "a") as f:
        f.write(
            f"{mask} {result_txt} | {[x.__class__.__name__ for x in pipeline._missing_subject_detectors]} | {[x.__class__.__name__ for x in pipeline._candidate_rankers]}\n")


if __name__ == "__main__":
    main()
