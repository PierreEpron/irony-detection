from transformers import Trainer
from torch import nn
import torch

class MCC_Loss(nn.Module):
    """
    From: https://github.com/kakumarabhishek/MCC-Loss
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc

    # def forward(self, inputs, targets):
    #     """
    #     MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
    #     where TP, TN, FP, and FN are elements in the confusion matrix.
    #     """
    #     tp = (inputs * targets).sum()
    #     tn = ((1 - inputs) * (1 - targets)).sum()
    #     fp = (inputs * (1 - targets)).sum()
    #     fn = ((1 - inputs) * targets).sum()

    #     numerator = tp * tn - fp * fn
    #     denominator = torch.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

    #     # Adding 1 to the denominator to avoid divide-by-zero errors.
    #     mcc = numerator / (denominator + 1.0)

    #     # print(mcc)

    #     return 1 - mcc
    
class IronyTrainer(Trainer):
    def __init__(self, loss_funcs, **kwargs):
        super().__init__(**kwargs)
        # Instantiate loss functions with their custom parameters
        self.loss_funcs = loss_funcs

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Forward
        outputs = model(**inputs)
        logits = nn.functional.sigmoid(outputs["logits"])
        # compue loss and sum their weight 
        loss = sum([f(logits[..., 1].float(), labels.float()) * w for f, w in self.loss_funcs])
        return (loss, outputs) if return_outputs else loss