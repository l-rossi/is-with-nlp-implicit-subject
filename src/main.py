from ImplicitSubjectPipeline import ImplicitSubjectPipeline
from candidate_extraction.CandidateExtracorImpl import CandidateExtractorImpl
from candidate_filtering.ImperativeFilter import ImperativeFilter
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
            PerplexityFilter(max_returned=10000),
            # , SemanticSimilarityFilter(), SubjectIfSameSentenceFilter(),
            # ProximityFilter()
        ],  # PerplexityRanker(),  #
        missing_subject_inserter=ImplicitSubjectInserterImpl(),
        verbose=True
    )

    n_inspected = 0
    n_correct = 0

    for source, target, gs in list(load_gold_standard())[0:50]:
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

        n_inspected += 1
        if gs.strip() == generated.strip():
            n_correct += 1

        print("-" * 4)
        print("Expected:", gs)
        print("Actual:  ", generated)
        print("-" * 9)

    print(f"Correct: {n_correct}/{n_inspected} ({n_correct / n_inspected * 100 :.2f}%)")


if __name__ == "__main__":
    main()
