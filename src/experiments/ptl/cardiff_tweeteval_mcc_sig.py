from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from transformers import AutoTokenizer
from lightning import Trainer
from pathlib import Path

from src.cls_ft import compute_mcc_loss, make_loader, CLSFineTuner
from src.model import cls_load_tweeteval
from src.utils import CustomWriter, get_plt_loggers, MonitoringMetrics


EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
PATIENCE = 5
MAX_LEN = 125
MODEL_NAME = "cardiffnlp/twitter-roberta-large-2022-154m"
RESULT_PATH = Path('results/cardiff_tweeteval_mcc')

monitor = MonitoringMetrics()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


data = cls_load_tweeteval({})


train_dataloader = make_loader(data[0][0], tokenizer, BATCH_SIZE, MAX_LEN, shuffle=True)
val_dataloader = make_loader(data[0][1], tokenizer, BATCH_SIZE, MAX_LEN, shuffle=False)
test_dataloader = make_loader(data[0][2], tokenizer, 1, MAX_LEN, shuffle=False)


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

# print(train_sample[0]['input_ids'].shape, train_sample[0]['attention_mask'].shape, )
# print(val_sample[0]['input_ids'].shape, val_sample[0]['attention_mask'].shape,)
# print(test_sample[0]['input_ids'].shape, test_sample[0]['attention_mask'].shape, )

# print(tokenizer.decode(train_sample[0]['input_ids'][0]))
# print(tokenizer.decode(test_sample[0]['input_ids'][0]))


model = CLSFineTuner(
    MODEL_NAME, 
    compute_mcc_loss, 
    learning_rate=LEARNING_RATE
)
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
# lr_finder = tuner.lr_find(model)
# new_lr = lr_finder.suggestion()

# monitor.set_size('lr_results', lr_finder.results)
# monitor.set_size('lr_suggestion', new_lr)

# model.hparams.lr = new_lr

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model.model.save_pretrained(RESULT_PATH)

monitor.set_time('training')

trainer.predict(model, test_dataloader, return_predictions=False)

monitor.set_time('predicting')

monitor.save(RESULT_PATH / 'monitoring.json')