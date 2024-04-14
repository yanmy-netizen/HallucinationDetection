import json
import torch
from utils import get_entailment_score, get_data

def get_NBC_features(model, tokenizer, data_path, classifier):
    """
    Get NBC features which are discretized based on entailment scores.

    Args:
        model: Entailment model.
        tokenizer: Tokenizer object.
        data_path (list): Positive and negative data_path.

    Returns:
        Dict[list, list]: Positive feature list and Negative feature list.
    """
    
    if classifier == None:
        pos_feature_list = [0] * 10
        neg_feature_list = [0] * 10
    else:
        pos_feature_list = [[0 for _ in range(10)] for _ in range(3)]
        neg_feature_list = [[0 for _ in range(10)] for _ in range(3)]
    
    pos_data = get_data(data_path[0])
    neg_data = get_data(data_path[1])
    
    for data in pos_data:
        premise = data['premise']
        hypothesis = data['hypothesis']
        score, _ = get_entailment_score(
            premise = premise,
            hypothesis = hypothesis,
            tokenizer = tokenizer,
            model = model,
        )
        if classifier != None:
            true_prob = classifier.predict([premise])[0]
            pos_feature_list[true_prob][int(score/10)] += 1
            pos_feature_list[2][int(score/10)] += 1
        else:
            pos_feature_list[int(score/10)] += 1
    
    for data in neg_data:
        premise = data['premise']
        hypothesis = data['hypothesis']
        score, _ = get_entailment_score(
            premise = premise,
            hypothesis = hypothesis,
            tokenizer = tokenizer,
            model = model,
        )
        if classifier != None:
            true_prob = classifier.predict([premise])[0]
            neg_feature_list[true_prob][int(score/10)] += 1
            neg_feature_list[2][int(score/10)] += 1
        else:
            neg_feature_list[int(score/10)] += 1

    return {
        "pos_features": pos_feature_list,
        "neg_features": neg_feature_list,
    }
