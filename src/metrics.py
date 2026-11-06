import numpy as np
import re
import string
from collections import Counter

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_metric(predictions, metric_name):
    if metric_name == "accuracy":
        return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])
    elif metric_name == "f1":
        return np.mean([f1(pred.predicted_candidate, pred.correct_candidate) for pred in predictions])


def f1(pred, gold):
    all_f1s = []
    for ans in gold:
        prediction_tokens = normalize_answer(pred).split()
        ground_truth_tokens = normalize_answer(ans).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            all_f1s.append(0)
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            all_f1s.append((2 * precision * recall) / (precision + recall))
    return np.max(all_f1s)
