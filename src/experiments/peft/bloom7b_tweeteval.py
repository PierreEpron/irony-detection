from peft import LoraConfig, TaskType
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from transformers import AutoTokenizer 
from lightning import Trainer
from pathlib import Path

from src.peft_ft import make_loader, CLMFineTuner
from src.model import cls_load_tweeteval
from src.utils import CustomWriter, MonitoringMetrics, get_plt_loggers, load_config


config = load_config()


EPOCHS = 50
BATCH_SIZE = 4
MODEL_NAME = "bigscience/bloom-7b1"
MAX_LEN = 125
PATIENCE = 5


PROMPT_TEMPLATE = Path('src/prompts/bloom_single_prompt.txt').read_text()
RESULT_PATH = Path('results/bloom7b_tweeteval')


peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.1)

monitor = MonitoringMetrics()
        
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=config['HF_TOKEN'])
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


data = cls_load_tweeteval({})


train_dataloader = make_loader(
    data[0][0], tokenizer, prompt_template=PROMPT_TEMPLATE, batch_size=BATCH_SIZE,
    max_len=MAX_LEN, train=True, shuffle=True)

val_dataloader = make_loader(
    data[0][1], tokenizer, prompt_template=PROMPT_TEMPLATE, batch_size=BATCH_SIZE,
    max_len=MAX_LEN, train=True, shuffle=False)

test_dataloader = make_loader(
    data[0][2], tokenizer, prompt_template=PROMPT_TEMPLATE, batch_size=1, 
    max_len=MAX_LEN, train=False, shuffle=False)


monitor.set_size('train', len(train_dataloader.dataset))
monitor.set_size('val', len(val_dataloader.dataset))
monitor.set_size('test', len(test_dataloader.dataset))


monitor.set_time('preprocessing')


# Uncomment to valid loaders
# train_sample = list(iter(train_dataloader))
# val_sample = list(iter(val_dataloader))
# test_sample = list(iter(test_dataloader))

# print(train_sample[0])
# print(val_sample[0])
# print(test_sample[0])

# print(train_sample[0]['input_ids'].shape, train_sample[0]['attention_mask'].shape, train_sample[0]['labels'].shape,)
# print(val_sample[0]['input_ids'].shape, val_sample[0]['attention_mask'].shape, val_sample[0]['labels'].shape,)
# print(test_sample[0]['input_ids'].shape, test_sample[0]['attention_mask'].shape, test_sample[0]['labels'].shape,)

# print(tokenizer.decode(train_sample[0]['input_ids'][0]))
# print(tokenizer.decode(test_sample[0]['input_ids'][0]))
# print(tokenizer.decode(test_sample[0]['labels'][0]))


finetuner = CLMFineTuner(MODEL_NAME, config['HF_TOKEN'], peft_config, tokenizer.eos_token_id)
pred_writer = CustomWriter(output_dir=RESULT_PATH, write_interval="epoch")


trainer = Trainer(
    default_root_dir=RESULT_PATH,
    max_epochs=EPOCHS, 
    log_every_n_steps=1, 
    logger=get_plt_loggers(RESULT_PATH, MODEL_NAME.split('/')[-1]),
    callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"), pred_writer],
    accelerator="gpu", devices=8, strategy="deepspeed_stage_2", precision=16,
)

# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(finetuner)
# new_lr = lr_finder.suggestion()

# monitor.set_size('lr_results', lr_finder.results)
# monitor.set_size('lr_suggestion', new_lr)

# finetuner.hparams.lr = new_lr

trainer.fit(model=finetuner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
finetuner.model.save_pretrained(RESULT_PATH)

monitor.set_time('training')

trainer.predict(finetuner, test_dataloader, return_predictions=False)

monitor.set_time('predicting')

monitor.save(RESULT_PATH / 'monitoring.json')

