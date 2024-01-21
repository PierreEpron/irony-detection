from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from datasets import Dataset
import torch

from pathlib import Path

from src.model import cls_load_tweeteval
from src.utils import write_jsonl

EPOCHS = 50
IDX_2_LABEL = {0:"no ironic", 1:"ironic"}
BATCH_SIZE = 16
MODEL_NAME = "bigscience/bloomz-560m"
MAX_LEN = 305

RESULT_PATH = Path('results/peft_test')

if not RESULT_PATH.is_dir():
    RESULT_PATH.mkdir()

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is ironic or not:",
    tokenizer_name_or_path=MODEL_NAME,
)

def tokenize(tokenizer, x, train=True):

    text_inputs = tokenizer(f"Tweet text : {x['text']} Label : ", add_special_tokens=False)
    label_inputs = tokenizer(IDX_2_LABEL[x['label']], add_special_tokens=False)    
    text_ids = [tokenizer.bos_token_id] + text_inputs['input_ids']
    label_ids = label_inputs['input_ids'] + [tokenizer.eos_token_id]


    if train == True: 
        input_ids = text_ids + label_ids
        return { 
            'input_ids': [tokenizer.pad_token_id] * (MAX_LEN - len(input_ids)) + input_ids,
            'attention_mask': [0] * (MAX_LEN - len(input_ids)) + [1] * len(input_ids),  
            "labels": [-100] * (MAX_LEN - len(label_ids)) + label_ids
        }
    else:
        return {
            'input_ids': text_ids,
            'attention_mask': [1] * len(text_ids),  
            "labels": label_ids
        }
        
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data = cls_load_tweeteval({})
train_set = Dataset.from_list(data[0][0]).map(lambda x: tokenize(tokenizer, x))
val_set = Dataset.from_list(data[0][1]).map(lambda x: tokenize(tokenizer, x))
test_set = Dataset.from_list(data[0][2]).map(lambda x: tokenize(tokenizer, x, train=False))

train_dataloader = DataLoader(train_set, shuffle=True, collate_fn=default_data_collator, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_set, shuffle=True, collate_fn=default_data_collator, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_set, shuffle=True, collate_fn=default_data_collator, batch_size=1)

# validate loaders
# batch_sample = list(iter(train_dataloader))
# batch_sample = list(iter(val_dataloader))
# test_sample = list(iter(test_dataloader))

# batch_sample = next(iter(train_dataloader))
# test_sample = next(iter(test_dataloader))

# print(batch_sample)
# print(test_sample)

class CLMFineTuner(LightningModule):
    def __init__(self, base_model_name, peft_config, eos_token_id, learning_rate=3e-2):
        super().__init__()
        self.peft_config = peft_config
        self.eos_token_id = eos_token_id
        self.learning_rate = learning_rate

        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = get_peft_model(self.model, peft_config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):    
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, batch_size=batch['input_ids'].shape[0], sync_dist=True)

    def predict_step(self, batch, batch_idx):
        outputs = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=10, eos_token_id=self.eos_token_id)
        return {
            'input_ids': batch['input_ids'].tolist()[0],
            'gold': batch['labels'].tolist()[0],
            'pred': outputs.tolist()[0],
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

tb_logger = TensorBoardLogger(RESULT_PATH / "tb_logs", name="peft")
csv_logger = CSVLogger(RESULT_PATH / "cv_logs", name="peft")

finetuner = CLMFineTuner(MODEL_NAME, peft_config, tokenizer.eos_token_id)

trainer = Trainer(
    default_root_dir=RESULT_PATH,
    max_epochs=EPOCHS, 
    log_every_n_steps=1, 
    logger=[tb_logger, csv_logger],
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
    accelerator="gpu", devices=8, strategy="deepspeed_stage_2", precision=16
)


trainer.fit(model=finetuner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

finetuner.model.save_pretrained(RESULT_PATH)

predictions = trainer.predict(finetuner, test_dataloader, ckpt_path='best')
write_jsonl(RESULT_PATH / 'predictions.jsonl', predictions)