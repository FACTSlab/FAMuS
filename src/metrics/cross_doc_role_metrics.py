import numpy as np
from typing import List, Dict, Optional

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
                    target_span,
                    edit_distance=edit_distance):
    """
    Compute the agreement score between two spans
    """
    if gold_span=="" and target_span=="":
        return 1
    gold_span_tokens = gold_span.split()
    target_span_tokens = target_span.split()
    len_gold = len(gold_span_tokens)
    len_target = len(target_span_tokens)

    distance = edit_distance(gold_span_tokens,
                             target_span_tokens) 

    denominator = (edit_distance._substitution_cost - 1)*min(len_gold, len_target) + \
                    max(len_gold, len_target)

    return 1 - (distance/denominator)


def tp_fp_fn_tn_role_agreement_multiple_gold(gold_values, 
                                             target_value):
    """
    Compute an agreement score between the gold dict and target role annotation
    dict.
    Note that the gold dict can have multiple values for a role. If the predicted
    role matches any of those values, it should be given a perfect agreement for that role

    Args:
        gold_dict: a dictionary of gold role annotations
            The gold dictionary will have the following format:
            {'Role1': ['Value1, 'Value2'], 'Role2': ['Value3'], 'Role3': ['Value4'], ...}
        target_dict: a dictionary of target role annotations
            The target dictionary will have the following format:
            {'Role': 'Value', 'Role': 'Value', 'Role': 'Value', ...}
        role_filler: if True, we will compute the agreement score for the role filler as well

    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count_tp_instances = 0

    # True positive
    if gold_values != set() and target_value != "":
        # compute tp for all roles present in predicted_dict
        max_tp_score = max([agreement_score(gold_value, 
                                            target_value) 
                            for gold_value in gold_values])
        tp += max_tp_score
        count_tp_instances += 1

        # divide the negation score equally between fp and fn
        negation = 1 - max_tp_score
        fp += negation/2
        fn += negation/2

    # False positive
    elif gold_values == set() and target_value != "":
        # for any non-empty role span that is not present in gold,
        # we increment a FP
        fp += 1

    # False negative
    elif gold_values != set() and target_value == "":
        # for any empty role span that is present in gold,
        # we increment a FN
        fn += 1
    
    # True negative
    elif gold_values == set() and target_value == "":
        tn += 1
    
    return tp, fp, fn, tn


def tp_fp_fn_tn_role_agreement_single_gold(gold_value, 
                                           target_value):
    """
    Compute an agreement score between the gold dict and target role annotation
    dict.
    Note that the gold dict can have multiple values for a role. If the predicted
    role matches any of those values, it should be given a perfect agreement for that role

    Args:
        gold_dict: a dictionary of gold role annotations
            The gold dictionary will have the following format:
            {'Role1': ['Value1, 'Value2'], 'Role2': ['Value3'], 'Role3': ['Value4'], ...}
        target_dict: a dictionary of target role annotations
            The target dictionary will have the following format:
            {'Role': 'Value', 'Role': 'Value', 'Role': 'Value', ...}
        role_filler: if True, we will compute the agreement score for the role filler as well

    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count_tp_instances = 0

    # True positive
    if gold_value != "" and target_value != "":
        # compute tp for all roles present in predicted_dict
        max_tp_score = agreement_score(gold_value, 
                                       target_value)
        tp += max_tp_score
        count_tp_instances += 1

        # divide the negation score equally between fp and fn
        negation = 1 - max_tp_score
        fp += negation/2
        fn += negation/2

    # False positive
    elif gold_value == "" and target_value != "":
        # for any non-empty role span that is not present in gold,
        # we increment a FP
        fp += 1

    # False negative
    elif gold_value != "" and target_value == "":
        # for any empty role span that is present in gold,
        # we increment a FN
        fn += 1
    
    # True negative
    elif gold_value == "" and target_value == "":
        tn += 1
    
    return tp, fp, fn, tn


def tp_fp_fn_tn_role_exact_match_multiple_gold(gold_values, 
                                               target_value):
    """
    Compute an agreement score between the gold dict and target role annotation
    dict.
    Note that the gold dict can have multiple values for a role. If the predicted
    role matches any of those values, it should be given a perfect agreement for that role

    Args:
        gold_dict: a dictionary of gold role annotations
            The gold dictionary will have the following format:
            {'Role1': ['Value1, 'Value2'], 'Role2': ['Value3'], 'Role3': ['Value4'], ...}
        target_dict: a dictionary of target role annotations
            The target dictionary will have the following format:
            {'Role': 'Value', 'Role': 'Value', 'Role': 'Value', ...}
        role_filler: if True, we will compute the agreement score for the role filler as well

    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count_tp_instances = 0

    # True positive
    if gold_values != set() and target_value != "":
        # compute tp for all roles present in predicted_dict
        max_tp_score = max([int(gold_value==target_value) 
                            for gold_value in gold_values])
        tp += max_tp_score
        count_tp_instances += 1

        # divide the negation score equally between fp and fn
        negation = 1 - max_tp_score
        fp += negation/2
        fn += negation/2

    # False positive
    elif gold_values == set() and target_value != "":
        # for any non-empty role span that is not present in gold,
        # we increment a FP
        fp += 1

    # False negative
    elif gold_values != set() and target_value == "":
        # for any empty role span that is present in gold,
        # we increment a FN
        fn += 1
    
    # True negative
    elif gold_values == set() and target_value == "":
        tn += 1
    
    return tp, fp, fn, tn


def tp_fp_fn_tn_role_exact_match_single_gold(gold_value, 
                                             target_value):
    """
    Compute an agreement score between the gold dict and target role annotation
    dict.
    Note that the gold dict can have multiple values for a role. If the predicted
    role matches any of those values, it should be given a perfect agreement for that role

    Args:
        gold_dict: a dictionary of gold role annotations
            The gold dictionary will have the following format:
            {'Role1': ['Value1, 'Value2'], 'Role2': ['Value3'], 'Role3': ['Value4'], ...}
        target_dict: a dictionary of target role annotations
            The target dictionary will have the following format:
            {'Role': 'Value', 'Role': 'Value', 'Role': 'Value', ...}
        role_filler: if True, we will compute the agreement score for the role filler as well

    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count_tp_instances = 0

    # True positive
    if gold_value != "" and target_value != "":
        # compute tp for all roles present in predicted_dict
        max_tp_score = int(gold_value==target_value)
        tp += max_tp_score
        count_tp_instances += 1

        # divide the negation score equally between fp and fn
        negation = 1 - max_tp_score
        fp += negation/2
        fn += negation/2

    # False positive
    elif gold_value == "" and target_value != "":
        # for any non-empty role span that is not present in gold,
        # we increment a FP
        fp += 1

    # False negative
    elif gold_value != "" and target_value == "":
        # for any empty role span that is present in gold,
        # we increment a FN
        fn += 1
    
    # True negative
    elif gold_value == "" and target_value == "":
        tn += 1
    
    return tp, fp, fn, tn