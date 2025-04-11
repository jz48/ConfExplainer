from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class BaseModel(torch.nn.Module):
    def __init__(self, training, input_type=None, task=None):
        """
        :param input_type: dense or sparse
        :param task: classification or regression
        :param training: train or eval
        """
        super(BaseModel, self).__init__()
        self.training = training
        self.input_type = input_type
        self.type = task
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def mse_loss(self, y_pred, y):
        return F.mse_loss(y_pred, y)

    def mape(self, y_pred, y):
        # print(y_pred, y)
        e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
        # print(e)
        return 100.0 * e

    def loss(self, y_pred, y):
        if self.type == 'reg':
            return self.mape(y_pred, y)
        elif self.type == 'cls':
            y_pred = y_pred.squeeze()
            return self.ce_loss(y_pred, y)
            # return self.mse_loss(y_pred, y)
