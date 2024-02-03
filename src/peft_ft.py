from transformers import default_data_collator
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from lightning import LightningModule
from peft import get_peft_model
from datasets import Dataset
import torch


IDX_2_LABEL = {0:"no ironic", 1:"ironic"}


def tokenize_example(tokenizer, example, template, idx2label=IDX_2_LABEL, train=True):
    ''' 
        Tokenize the given example using the given "template".
        Used for all the peft experiments.

        Parameters
        ==========
        tokenizer: Tokenizer used by the model. 
        Should have this a least this set of special tokens {bos_token_id, pad_token_id, eos_token_id}.
        Should have a __call__ function for tokenization.
        
        example: The target example to tokenize. 
        A dict with at least the following keys {'text', 'label'}.
        
        template: The template to use before tokenization.
        Will be use like this: template.format(**example).
        Should at least contain a text argument.

        idx2label (default=IDX_2_LABEL): A dict used to convert label ids to text

        train (default=True): IF true, label will be added to input ids.

        Returns
        =======
        A dict containing:
        'input_ids': A list of token ids from the concatenation of given example text and label.
        'attention_mask': A list of "1" of the same size than input_ids.
        'labels': A list of token ids. Text tokens are set to -100. Only label token ids are set with their value.
    '''

    text_inputs = tokenizer(template.format(**example), add_special_tokens=False)
    label_inputs = tokenizer(idx2label[example['label']], add_special_tokens=False)    
    text_ids = [tokenizer.bos_token_id] + text_inputs['input_ids']
    label_ids = label_inputs['input_ids'] + [tokenizer.eos_token_id]

    if train == True: 
        input_ids = text_ids + label_ids
        return { 
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),  
            "labels": [-100] * (len(input_ids) - len(label_ids)) + label_ids
        }
    else:
        return {
            'input_ids': text_ids,
            'attention_mask': [1] * len(text_ids),  
            "labels": label_ids
        }


def pad_example(tokenizer, example, max_len=100):
    '''
        Pad from the left the given tokenized "example".
        Padded to "max_len".
    '''
    pad_size = max_len - len(example['input_ids'])
    return { 
        'input_ids': [tokenizer.pad_token_id] * pad_size + example['input_ids'],
        'attention_mask': [1] * pad_size + example['attention_mask'],  
        "labels": [-100] * pad_size + example['labels']
    }


def make_loader(
        data, 
        tokenizer, 
        prompt_template, 
        batch_size, 
        max_len, 
        idx2label=IDX_2_LABEL, 
        train=True, 
        shuffle=True
    ):
    '''
        Create dataset, tokenize examples, filter example by max_len, pad examples then return a loader.
    '''
    data_set = Dataset.from_list(data).map(lambda x: tokenize_example(tokenizer, x, prompt_template, idx2label=idx2label, train=train))
    data_set = data_set.filter(lambda x: len(x['input_ids']) <= max_len)

    if train:
        data_set = data_set.map(lambda x: pad_example(tokenizer, x, max_len))

    return DataLoader(data_set, collate_fn=default_data_collator, batch_size=batch_size, shuffle=shuffle)


class CLMFineTuner(LightningModule):
    '''
        Finetuner used for peft experiments.
    '''

    def __init__(self, base_model_name, hf_token, peft_config, eos_token_id, learning_rate=3e-2):
        super().__init__()
        self.peft_config = peft_config
        self.eos_token_id = eos_token_id
        self.learning_rate = learning_rate

        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, token=hf_token, device_map={'': 0} ) # Try for fix cuda issue
        self.model = get_peft_model(self.model, peft_config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, batch_size=batch['input_ids'].shape[0], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):    
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, batch_size=batch['input_ids'].shape[0], on_step=False, on_epoch=True, sync_dist=True)

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