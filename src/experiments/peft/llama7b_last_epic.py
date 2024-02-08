from dataclasses import dataclass, field
from typing import Optional


from transformers import (
    AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, 
    AutoTokenizer, EarlyStoppingCallback
)

from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from tqdm import tqdm
import torch

from src.prompt import generate_turns, load_phrases
from src.utils import load_config, write_jsonl
from src.model import cls_load_epic


config = load_config()

model_name = "meta-llama/Llama-2-7b-chat-hf"

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=model_name, metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=125, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(
        default="results/llama7b_last_epic", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=50, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

    
    ##### Added args #####
    phrases_path: Optional[str] = field(default="src/prompts/double_phrases_train.json", metadata="path to find phrases used to build the prompt for training")
    early_stopping_patience: Optional[int] = field(default=5, metadata="stop training when the specified metric worsens for early_stopping_patience evaluation calls")
    early_stopping_threshold: Optional[float] = field(default=0.0, metadata="how much the specified metric must improve to satisfy early stopping conditions.")
    do_eval: Optional[bool] = field(default=True, metadata="whether to run evaluation on the validation set or not.")
    evaluation_strategy: Optional[str] = field(default="epoch", metadata="The evaluation strategy to adopt during training.")
    load_best_model_at_end: Optional[bool] = field(default=True, metadata="whether or not to load the best model found during training at the end of training.")
    save_strategy: Optional[str] = field(default="epoch", metadata="The checkpoint save strategy to adopt during training.")

script_args = ScriptArguments()

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, token=config['HF_TOKEN'])
tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
tokenizer.use_default_system_prompt = False


def preprocess(item, phrases):

    item['text_label'] = phrases['labels'][0]['values'][(item['label'])]
    turns, _, _ = generate_turns(item, phrases)
    input_ids = tokenizer.apply_chat_template(turns)
    
    if len(input_ids) > script_args.seq_length:
        return None

    return {
        'id_original': item['id_original'],
        'text': tokenizer.decode(input_ids),
        'label': item['label'],
        'text_label': item['label'],
    }


train_data, val_data, test_data = list(cls_load_epic(config))[0]

train_phrases, labels = load_phrases(script_args.phrases_path)

train_data = [preprocess(item, train_phrases) for item in train_data]
val_data = [preprocess(item, train_phrases) for item in val_data]
test_data = [preprocess(item, train_phrases) for item in test_data]


train_set = Dataset.from_list([item for item in train_data if item])
val_set = Dataset.from_list([item for item in val_data if item])
test_set = Dataset.from_list([item for item in test_data if item])

print(f'{len(train_set)/len(train_data)}, {len(val_set)/len(val_data)}, {len(test_set)/len(test_data)}')

print(train_set[0]['text'])

quantization_config = BitsAndBytesConfig(
    load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
)
torch_dtype = torch.bfloat16
device_map = {"": 0}


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    do_eval=script_args.do_eval,
    evaluation_strategy=script_args.evaluation_strategy,
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    load_best_model_at_end=script_args.load_best_model_at_end,
    save_strategy=script_args.save_strategy,
)


peft_config = LoraConfig(
    r=script_args.peft_lora_r,
    lora_alpha=script_args.peft_lora_alpha,
    bias="none",
    task_type="CAUSAL_LM",
)


model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    token=config['HF_TOKEN']
)

model.config.pad_token_id = tokenizer.pad_token_id

early_stop = EarlyStoppingCallback(
    script_args.early_stopping_patience,
    script_args.early_stopping_threshold
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=train_set,
    eval_dataset=val_set,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    callbacks=[early_stop]
)


trainer.train()
trainer.save_model(training_args.output_dir)
trainer.save_state()

model = AutoModelForCausalLM.from_pretrained(
    script_args.output_dir,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    token=config['HF_TOKEN']
)

results = []
softmax = torch.nn.Softmax(dim=-1)

label_ids = tokenizer.encode(labels, add_special_tokens=False, return_tensors='pt')[0].to(model.device)

for item in tqdm(test_set, 'Test loop:'):

    input_ids = tokenizer.encode(item['text'], return_tensors='pt')[..., :-3].to(model.device)
    logits = model(input_ids).logits

    full_scores = softmax(logits[..., -1, :])
    full_scores_labels = full_scores[..., label_ids].detach().cpu().numpy()
    scores = softmax(logits[..., -1, label_ids]).detach().cpu().numpy()

    results.append({
        'id_original': item['id_original'],
        'prompt':item['text'], 
        'gold':item['label'],

        'full_scores': full_scores_labels[0].tolist(),
        'scores': scores[0].tolist(),

        'full_pred': int(full_scores.detach().cpu().numpy().argmax()),
        'full_pred_labels': int(full_scores_labels.argmax()),
        'pred': int(scores.argmax()),
    })

write_jsonl(script_args.output_dir + "/predictions.jsonl", results)
