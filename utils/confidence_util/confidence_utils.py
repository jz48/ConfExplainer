import numpy as np
import json
import os


def save_confidence_results(score, expls, save_path, file_name):
    # [s, auc_score, rdd_score, nll_score], explanations, save_path=folder_path, file_name=file_name
    # print(expls[0])
    explanations = [[i[0].item(), i[1].detach().numpy().tolist(), np.array(i[2]).tolist(),
                     i[3].detach().numpy().tolist()] for i in expls]
    score.append(explanations)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = os.path.join(save_path, file_name)
    with open(path, 'w') as f:
        f.write(json.dumps(score))
    pass


def load_confidence_results(save_path, file_name):
    path = os.path.join(save_path, file_name)
    with open(path, 'r') as f:
        data = json.loads(f.readline())
    return data


