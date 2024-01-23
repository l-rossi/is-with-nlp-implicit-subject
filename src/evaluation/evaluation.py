import math
from collections import defaultdict
from typing import List


class StatAcc:
    """
    Accumulates stats
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def apply(self, acc: "StatAcc"):
        """
        Adds the stats from one accumulator to this one.
        """
        self.tp += acc.tp
        self.tn += acc.tn
        self.fp += acc.fp
        self.fn += acc.fn

    def precision(self) -> float:
        """
        Returns the current precision.
        """
        if self.tp + self.fp == 0:
            return math.nan
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        """
        Returns the current recall.
        """
        if self.tp + self.fp == 0:
            return math.nan
        return self.tp / (self.tp + self.fn)

    def __str__(self):
        return f"StatAcc(tp = {self.tp}, fp = {self.fp}, tn = {self.tn}, fn = {self.fn})"


def evaluate_detection(expected: List[str], actual: List[str]) -> StatAcc:
    """
    Evaluates a list of detected subjects against a gold standard
    """

    acc = StatAcc()

    exp_grouping = defaultdict(int)
    act_grouping = defaultdict(int)

    for e in expected:
        exp_grouping[e] += 1

    for a in actual:
        act_grouping[a] += 1

    for k in exp_grouping.keys() | act_grouping.keys():
        exp = exp_grouping[k]
        act = act_grouping[k]
        acc.tp += min(exp, act)
        acc.fp += max(0, act - exp)
        acc.fn += max(0, exp - act)

    return acc
