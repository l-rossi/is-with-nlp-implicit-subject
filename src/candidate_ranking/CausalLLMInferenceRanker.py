from typing import List

import numpy as np
from spacy.tokens import Token
from transformers import AutoModelForCausalLM, GPT2Tokenizer

from candidate_ranking.CandidateRanker import CandidateRanker


class CausalLLMInferenceRanker(CandidateRanker):
    """
    TODO delete me
    """

    def __init__(self, model_id="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def rank(self, target: Token, candidates: List[Token]) -> List[Token]:

        # This is dumb....

        candidate_spans = [x.doc[x.left_edge.i: x.right_edge.i + 1].text for x in candidates]
        inputs = self.tokenizer([f"{target.text} by "], return_tensors="pt")

        for cs in candidate_spans:
            print(str(cs))
            cs_tokens = self.tokenizer(str(cs))["input_ids"]
            outputs = self.model.generate(**inputs, max_new_tokens=8, return_dict_in_generate=True, output_scores=True,
                                          num_beams=5, force_words_ids=[cs_tokens])

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]

            generated_tokens = outputs.sequences[:, input_length:]

            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                # | token | token string | logits | probability
                print(
                    f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

        return []
