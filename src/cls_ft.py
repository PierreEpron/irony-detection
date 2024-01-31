from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from lightning import LightningModule
from datasets import Dataset
import torch


class DataCollatorWithPadding:
    ''' Collate data and pad them from the right'''

    def __init__(self, pad_token) -> None:
        self.pad_token = pad_token

    def __call__(self, batch):
        return self.collate_inputs(batch) | self.collate_label(batch) | self.collate_others(batch)

    def collate_inputs(self, batch):
        '''Collate and pad "input_ids" and "attention_mask"'''
        outputs = {}

        # Compute maximum len of input_ids inside the batch
        max_len = max([len(item['input_ids']) for item in batch])
        
        outputs['input_ids'] = self.make_padded_tensor([item['input_ids'] for item in batch], max_len, self.pad_token)
        outputs['attention_mask'] = self.make_padded_tensor([item['attention_mask'] for item in batch], max_len, 0)

        return outputs
    
    def make_padded_tensor(self, batch, max_len, pad_value):
        ''' Make a padded tensor for the given "batch" of item'''
        t = torch.tensor([self.pad_inputs(item, max_len, pad_value) for item in batch]).squeeze()
        if len(t.shape) == 1:
            t = t.unsqueeze(dim=0)
        return t

    def pad_inputs(self, item, max_len, pad_value):
        ''' Pad the given "item" to the given "max_len" with the given "pad_value" '''
        return [item + [pad_value] * (max_len - len(item))]

    def collate_label(self, batch):
        return {'label': torch.tensor([item['label'] for item in batch])}

    def collate_others(self, batch):
        '''Collate all other inputs than "input_ids' and "attention_mask" '''
        outputs = {}

        # Get other keys than "input_ids", "attention_mask" and "label"
        other_keys = set(batch[0].keys()) - { 'input_ids', 'attention_mask', 'label' }

        # Collate them
        for k in other_keys:
            outputs[k] = [item[k] for item in batch]

        return outputs


def make_loader(data, tokenizer, batch_size, max_len, shuffle=True):
    '''
        Create dataset, tokenize examples, filter example by max_len, pad examples then return a loader.
    '''
    data_set = Dataset.from_list(data).map(lambda x: tokenizer(x['text']))
    data_set = data_set.filter(lambda x: len(x['input_ids']) <= max_len)
    return DataLoader(data_set, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer.pad_token_id), shuffle=shuffle)


class CLSSigmoidFineTuner(LightningModule):
    '''
        Finetuner for cls task
    '''
    def __init__(self, base_model_name, loss_func, learning_rate):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, output_attentions=True, num_labels=1)
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs['logits'] = torch.sigmoid(outputs["logits"])
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])

        loss = self.loss_func(outputs['logits'], batch['label'].long())
        self.log("train_loss", loss, batch_size=1, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):    
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])

        val_loss = self.loss_func(outputs['logits'], batch['label'].long())
        self.log("val_loss", val_loss, batch_size=1, on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])
        return {
            'id_original':batch['id_original'][0], 
            'text':batch['text'][0], 
            'gold':batch['label'].item(), 
            'pred':(outputs['logits'] > .5).int().item(), 
            'score':outputs['logits'].tolist()[0]
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class CLSFineTuner(LightningModule):
    '''
        Finetuner for cls task
    '''
    def __init__(self, base_model_name, loss_func, learning_rate):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, output_attentions=True, num_labels=2)
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs['logits'] = torch.softmax(outputs["logits"], dim=-1)
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])

        loss = self.loss_func(outputs['logits'], batch['label'].long())
        self.log("train_loss", loss, batch_size=1, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):    
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])

        val_loss = self.loss_func(outputs['logits'], batch['label'].long())
        self.log("val_loss", val_loss, batch_size=1, on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])
        return {
            'id_original':batch['id_original'][0], 
            'text':batch['text'][0], 
            'gold':batch['label'].item(), 
            'pred':outputs['logits'].argmax().item(), 
            'score':outputs['logits'].tolist()[0]
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def compute_2d_mcc_loss(x, y):
    ''' Compute mcc loss on each logits and average them '''
    mcc = MCCLoss()
    return (mcc(x[..., 0], 1 - y.float()) + mcc(x[..., 1], y.float())) / 2
    # return mcc(x[..., 1], y)

def compute_1d_mcc_loss(x, y):
    ''' Compute mcc loss on each logits and average them '''
    mcc = MCCLoss()
    return mcc(x, y)


class MCCLoss(torch.nn.Module):
    """
    From: https://github.com/kakumarabhishek/MCC-Loss
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCCLoss, self).__init__()

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