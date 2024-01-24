import spacy
from dotenv import load_dotenv

from ImplicitSubjectPipeline import ImplicitSubjectPipeline
from candidate_extraction.CandidateExtracorImpl import CandidateExtractorImpl
from candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from candidate_filtering.ImperativeFilter import ImperativeFilter
from candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from candidate_filtering.PerplexityFilter import PerplexityFilter
from evaluation.evaluation import ClassificationStatisticsAccumulator, evaluate_detection, FilterFailAccumulator
from insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from missing_subject_detection.GerundDetector import GerundDetector
from missing_subject_detection.ImperativeDetector import ImperativeDetector
from missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from missing_subject_detection.PassiveDetector import PassiveDetector
from util import load_gold_standard, dependency_trees_equal

load_dotenv()


def main():
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[PassiveDetector(), ImperativeDetector(), GerundDetector(),
                                   NominalizedGerundWordlistDetector()],
        candidate_extractor=CandidateExtractorImpl(),
        candidate_rankers=[
            ImperativeFilter(),
            PartOfSpeechFilter(),
            DependentOfSameSentenceFilter(),
            # ChatGPTFilter(),
            # SimilarityFilter(use_context=True, model="en_use_lg"),
            PerplexityFilter(max_returned=10000),
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

    detection_accumulator = ClassificationStatisticsAccumulator()
    filter_stats_accumulator = FilterFailAccumulator()

    mask = ""
    for i, (source, inp, gs, impl_subjects, targets) in enumerate(list(load_gold_standard())[:]):

        print(f"Enter {i}")
        print("Context:")
        print(source)
        print("-" * 5)
        print("Inspected text:")
        print(inp)
        print("-" * 5)

        generated = pipeline.apply(
            inspected_text=inp,
            context=source
        )

        current_stats = evaluate_detection(targets, [x.token.text for x in pipeline.last_detections()])

        detection_accumulator.apply(current_stats)
        filter_stats_accumulator.apply(pipeline.last_filter_log(), targets, impl_subjects)

        print(
            f"Detection stats: Precision {current_stats.precision() * 100 :.2f}%, Recall {current_stats.recall() * 100 :.2f}%")

        gs_doc = similarity_nlp(gs)
        generated_doc = similarity_nlp(generated)
        similarity = gs_doc.similarity(generated_doc)
        n_inspected += 1

        if gs.strip() == generated.strip():  # or similarity > 0.999:  # <- seems to be a good cutoff for allowing minor permutations in the sentence structure
            n_correct += 1
            mask += "x"
        elif dependency_trees_equal(gs_doc, generated_doc):
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
        print(f"dep equality: {dependency_trees_equal(gs_doc, generated_doc)}")

        print("Filter failures by filter:", filter_stats_accumulator.counts())
        print(f"Correctly filtered: {filter_stats_accumulator.performance_str()}")

        print("-" * 9)

    print(mask)
    result_txt = f"Correct: {n_correct}/{n_inspected} ({n_correct / n_inspected * 100 :.2f}%). Detection stats: Precision {detection_accumulator.precision() * 100 :.2f}%, Recall {detection_accumulator.recall() * 100 :.2f}%"
    print(result_txt)
    with open("./log/res", "a") as f:
        f.write(
            f"{mask} {result_txt} | {[x.__class__.__name__ for x in pipeline._missing_subject_detectors]} | {[x.__class__.__name__ for x in pipeline._candidate_rankers]}\n")


if __name__ == "__main__":
    main()
