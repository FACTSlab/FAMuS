import numpy as np
from typing import List, Dict, Optional, Set

class EditDistance:
    '''Distance between strings

    Parameters
    ----------
    insertion_cost
    deletion_cost
    substitution_cost
    '''
    def __init__(self, insertion_cost: float = 1., deletion_cost: float = 1., substitution_cost: Optional[float] = None):
        self._insertion_cost = insertion_cost
        self._deletion_cost = deletion_cost

        if substitution_cost is None:
            self._substitution_cost = insertion_cost + deletion_cost
        else:
            self._substitution_cost = substitution_cost

    def __call__(self, source: List[str], target: List[str]) -> float:
        n, m = len(source), len(target)
        source, target = ['#']+source, ['#']+target

        distance = np.zeros([n+1, m+1], dtype=float)

        for i in range(1,n+1):
            distance[i,0] = distance[i-1,0]+self._deletion_cost

        for j in range(1,m+1):
            distance[0,j] = distance[0,j-1]+self._insertion_cost

        for i in range(1,n+1):
            for j in range(1,m+1):
                if source[i] == target[j]:
                    substitution_cost = 0.
                else:
                    substitution_cost = self._substitution_cost

                costs = np.array([distance[i-1,j]+self._deletion_cost,
                                  distance[i-1,j-1]+substitution_cost,
                                  distance[i,j-1]+self._insertion_cost])

                distance[i,j] = costs.min()

        return distance[n,m]

edit_distance = EditDistance(insertion_cost=1,
                             deletion_cost=1,
                             substitution_cost=2)

def agreement_score(gold_span, 
                    predicted_span,
                    edit_distance=edit_distance):
    """
    Compute the agreement score between two spans
    """
    if gold_span=="" and predicted_span=="":
        return 1
    gold_span_tokens = gold_span.split()
    predicted_span_tokens = predicted_span.split()
    len_gold = len(gold_span_tokens)
    len_target = len(predicted_span_tokens)

    distance = edit_distance(gold_span_tokens,
                             predicted_span_tokens) 

    denominator = (edit_distance._substitution_cost - 1)*min(len_gold, len_target) + \
                    max(len_gold, len_target)

    return 1 - (distance/denominator)


def exact_match_score(gold_span,
                        predicted_span):
    """
    Compute the agreement score between two spans using exact match
    """
    return int(gold_span==predicted_span)


def tp_fp_fn_tn_role_agreement_multiple_gold(gold_spans: Set[str], 
                                             predicted_span: str,
                                             agreement_fn: callable=agreement_score):
    """
    Given a predicted value and a list of gold values for some role,
    Compute the tp, fp, fn, tn scores. 

    Args:
        gold_spans: a set of textual spans corresponding to the gold
                    role fillers for some role. The multiple gold 
                    spans generally come from a coref cluster
        target_value: a single textual span corresponding to the predicted
                     role filler for the same role.
        agreement_fn: a function that computes the agreement score between
                        two spans. The default is the edit distance function.
                        We can also use exact match.

    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count_tp_instances = 0

    # True positive
    if gold_spans != set() and predicted_span != "":
        # compute tp for all roles present in predicted_dict
        max_tp_score = max([agreement_fn(gold_span, 
                                         predicted_span) 
                            for gold_span in gold_spans])
        tp += max_tp_score
        count_tp_instances += 1

        # divide the negation score equally between fp and fn
        negation = 1 - max_tp_score
        fp += negation/2
        fn += negation/2

    # False positive
    elif gold_spans == set() and predicted_span != "":
        # for any non-empty role span that is not present in gold,
        # we increment a FP
        fp += 1

    # False negative
    elif gold_spans != set() and predicted_span == "":
        # for any empty role span that is present in gold,
        # we increment a FN
        fn += 1
    
    # True negative
    elif gold_spans == set() and predicted_span == "":
        tn += 1
    
    return tp, fp, fn, tn


def tp_fp_fn_tn_role_agreement_single_gold(gold_span: str,
                                           predicted_span: str,
                                           agreement_fn: callable=agreement_score):
    """
    Given a predicted value and a gold value for some role,
    Compute the tp, fp, fn, tn scores.

    Args:
        gold_span: a single textual span corresponding to the gold
                    role filler for some role. 
        target_value: a single textual span corresponding to the predicted
                     role filler for the same role.
        agreement_fn: a function that computes the agreement score between
                        two spans. The default is the edit distance function.
                        We can also use exact match.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count_tp_instances = 0

    # True positive
    if gold_span != "" and predicted_span != "":
        # compute tp for all roles present in predicted_dict
        max_tp_score = agreement_fn(gold_span, 
                                    predicted_span)
        tp += max_tp_score
        count_tp_instances += 1

        # divide the negation score equally between fp and fn
        negation = 1 - max_tp_score
        fp += negation/2
        fn += negation/2

    # False positive
    elif gold_span == "" and predicted_span != "":
        # for any non-empty role span that is not present in gold,
        # we increment a FP
        fp += 1

    # False negative
    elif gold_span != "" and predicted_span == "":
        # for any empty role span that is present in gold,
        # we increment a FN
        fn += 1

    # True negative
    elif gold_span == "" and predicted_span == "":
        tn += 1

    return tp, fp, fn, tn
