from ImplicitSubjectPipeline import ImplicitSubjectPipeline
from candidate_extraction.CandidateExtracorImpl import CandidateExtractorImpl
from candidate_ranking.ProximityRanker import ProximityRanker
from insertion.ImplcitSubjectInserterImpl import ImplicitSubjectInserterImpl
from missing_subject_detection.ImperativeDetector import ImperativeDetector
from missing_subject_detection.PassiveDetector import PassiveDetector
from util import load_gold_standard


def main():
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[PassiveDetector(), ImperativeDetector()],
        candidate_extractor=CandidateExtractorImpl(),
        candidate_ranker=ProximityRanker(),  # PerplexityRanker(),  #
        missing_subject_inserter=ImplicitSubjectInserterImpl(),
        verbose=True
    )

    n_inspected = 0
    n_correct = 0

    for source, target, gs in list(load_gold_standard())[:1]:
        print("Context:")
        print(source)
        print("-" * 5)
        print("Inspected text:")
        print(target)
        print("-" * 4)

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
