
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from lightning import LightningModule, Trainer

from torch.utils.data import DataLoader
import torch

from pathlib import Path

from src.tokenizer import cls_single_tokenize
from src.model import cls_load_tweeteval
from src.utils import write_jsonl
from src.training import MCC_Loss

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
RESULT_PATH = Path('results/plt_bce')

if not RESULT_PATH.is_dir():
    RESULT_PATH.mkdir()

class DataCollatorWithPadding:
    def __init__(self, pad_token) -> None:
        self.pad_token = pad_token

    def __call__(self, batch):
        return self.collate_inputs(batch) | self.collate_label(batch)| self.collate_others(batch)

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
        other_keys = set(batch[0].keys()) - { 'input_ids', 'attention_mask', 'label'}

        # Collate them
        for k in other_keys:
            outputs[k] = [item[k] for item in batch]

        return outputs

class IronyDetectionFineTuner(LightningModule):
    def __init__(self, base_model_name, loss_func, sec_loss_func, learning_rate):
        super().__init__()
        # self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, output_attentions=True, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, output_attentions=True, num_labels=2)
        self.loss_func = loss_func
        self.sec_loss_func = sec_loss_func
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs['logits'] = torch.sigmoid(outputs["logits"])
        outputs['logits'] = torch.softmax(outputs["logits"], dim=-1)
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])

        loss = self.loss_func(outputs['logits'][..., 1], batch['label'].float())
        sec_loss = self.sec_loss_func(outputs['logits'][..., 1], batch['label'].float())

        self.log("train_loss", loss, batch_size=1, sync_dist=True)
        self.log("sec_train_loss", sec_loss, batch_size=1, sync_dist=True)
        self.log("train_mean_logits", outputs['logits'][..., 1].detach().mean(), batch_size=1, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):    
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])

        val_loss = self.loss_func(outputs['logits'][..., 1], batch['label'].float())
        sec_val_loss = self.sec_loss_func(outputs['logits'][..., 1], batch['label'].float())

        self.log("val_loss", val_loss, batch_size=1, sync_dist=True)
        self.log("sec_val_loss", sec_val_loss, batch_size=1, sync_dist=True)
        self.log("val_mean_logits", outputs['logits'][..., 1].detach().mean(), batch_size=1, sync_dist=True)


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

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m")

data = cls_load_tweeteval({})
train_set = Dataset.from_list(data[0][0]).map(lambda x: cls_single_tokenize(tokenizer, x))
val_set = Dataset.from_list(data[0][1]).map(lambda x: cls_single_tokenize(tokenizer, x))
test_set = Dataset.from_list(data[0][2]).map(lambda x: cls_single_tokenize(tokenizer, x))

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=DataCollatorWithPadding(tokenizer.pad_token_id), shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=DataCollatorWithPadding(tokenizer.pad_token_id))
test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=DataCollatorWithPadding(tokenizer.pad_token_id))

model = IronyDetectionFineTuner(
    'cardiffnlp/twitter-roberta-large-2022-154m', 
    torch.nn.BCEWithLogitsLoss(),
    MCC_Loss(), 
    learning_rate=LEARNING_RATE
)

# tb_logger = TensorBoardLogger(RESULT_PATH / "tb_logs", name="bce")
# csv_logger = CSVLogger(RESULT_PATH / "cv_logs", name="bce")

# trainer = Trainer(
#     default_root_dir=RESULT_PATH,
#     max_epochs=EPOCHS, 
#     log_every_n_steps=1, 
#     logger=[tb_logger, csv_logger],
#     callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
#     accelerator="gpu", devices=8, strategy="deepspeed_stage_2", precision=16
# )

# trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# model.model.save_pretrained(RESULT_PATH)

trainer = Trainer(
    default_root_dir=RESULT_PATH,
    accelerator="gpu", devices=1, precision=16
)

predictions = trainer.predict(model, test_dataloader)
write_jsonl(RESULT_PATH / 'predictions.jsonl', predictions)