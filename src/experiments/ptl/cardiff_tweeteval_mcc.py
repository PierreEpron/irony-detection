from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer
from lightning import Trainer
from pathlib import Path

from src.cls_ft import compute_mcc_loss, make_loader, CLSFineTuner
from src.model import cls_load_tweeteval
from utils import get_plt_loggers


EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
PATIENCE = 5
MODEL_NAME = "cardiffnlp/twitter-roberta-large-2022-154m"
RESULT_PATH = Path('results/cardiff_tweeteval_mcc')


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


data = cls_load_tweeteval({})


train_dataloader = make_loader(data[0][0], tokenizer, BATCH_SIZE, shuffle=True)

val_dataloader = make_loader(data[0][1], tokenizer, BATCH_SIZE, shuffle=False)

test_dataloader = make_loader(data[0][2], tokenizer, 1, shuffle=False)


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


trainer = Trainer(
    default_root_dir=RESULT_PATH,
    max_epochs=EPOCHS, 
    log_every_n_steps=1, 
    logger=get_plt_loggers(RESULT_PATH, MODEL_NAME.split('/')[-1]),
    callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")],
    accelerator="gpu", devices=8, strategy="deepspeed_stage_2", precision=16
)


trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model.model.save_pretrained(RESULT_PATH)
trainer.predict(model, test_dataloader)