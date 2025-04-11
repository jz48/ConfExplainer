# import ...

class ConfidenceEstimator(object):
    def __init__(self):
        pass

    def compute_confidence(self, gt, expl):
        '''
        :input:
        gt: ground truths of original graphs' explanations [N, 1, E], N is number of graph, E is number of edges
        expl: the predicted explanations [N, 1, E]
        :return:
        confidence: [N, 1, 1]
        '''

        for i in range(len(gt)):
            # todo
            for edge in gt[i]:
                # edge_confidence = - (boolean(y^*==y) * |exp[i]-gt[i]|/(gt[i]+bias))
                #
                pass
            # gt_confidence = avg(edge_confidences)
            pass
