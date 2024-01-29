from dotenv import load_dotenv

import evaluation
from ImplicitSubjectPipeline import ImplicitSubjectPipeline
from candidate_extraction.CandidateExtracorImpl import CandidateExtractorImpl
from candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from candidate_filtering.ChatGPTFilter import ChatGPTFilter
from candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from candidate_filtering.ImperativeFilter import ImperativeFilter
from candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from candidate_filtering.PerplexityFilter import PerplexityFilter
from candidate_filtering.PreviouslyMentionedRelationFilter import PreviouslyMentionedRelationFilter
from candidate_filtering.ProximityFilter import ProximityFilter
from candidate_filtering.SimilarityFilter import SimilarityFilter
from evaluation.util import run_gs_eval
from insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from missing_subject_detection.GerundDetector import GerundDetector
from missing_subject_detection.ImperativeDetector import ImperativeDetector
from missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from missing_subject_detection.PassiveDetector import PassiveDetector

load_dotenv()


def main():
    """
    Creates an ImplicitSubjectPipeline and runs it against the gold standard.
    """

    # setup the used pipeline here
    # Feel free to comment or uncomment components in either the missing_subject_detectors or
    # the candidate_filters. The current selection provides the best results for the gold standard.
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[
            PassiveDetector(),
            ImperativeDetector(),
            GerundDetector(),
            NominalizedGerundWordlistDetector(),
            # NounVerbStemDetector(),
        ],
        candidate_filters=[
            ImperativeFilter(),
            PartOfSpeechFilter(),
            DependentOfSameSentenceFilter(),
            # ChatGPTFilter(),
            # SimilarityFilter(use_context=True, model="en_use_lg"),
            PerplexityFilter(max_returned=10000),
            CandidateTextOccurrenceFilter(),
            # PreviouslyMentionedRelationFilter(),
            # ProximityFilter(),
        ],
        verbose=True
    )

    # The pipeline can be used by simply providing a text to be inspected for implicit subjects and
    # a context from which candidates should be taken. (Note, that ideally the inspected text should be contained
    # in the context for all filters to work correctly.)
    # context = """
    #   Ozymandias
    #   BY PERCY SHELLEY
    #
    #   I met a traveller from an antique land,
    #   Who said—"Two vast and trunkless legs of stone
    #   Stand in the desert... Near them, on the sand,
    #   Half sunk a shattered visage lies, whose frown,
    #   And wrinkled lip, and sneer of cold command,
    #   Tell that its sculptor well those passions read
    #   Which yet survive, stamped on these lifeless things,
    #   The hand that mocked them, and the heart that fed;
    #   And on the pedestal, these words appear:
    #   My name is Ozymandias, King of Kings;
    #   Look on my Works, ye Mighty, and despair!
    #   Nothing beside remains. Round the decay
    #   Of that colossal Wreck, boundless and bare
    #   The lone and level sands stretch far away."
    # """
    # result = pipeline.apply("Mocked, I made my retreat.", context)
    # print(result)

    # runs the evaluation on the gold standard
    run_gs_eval(pipeline)


if __name__ == "__main__":
    main()
