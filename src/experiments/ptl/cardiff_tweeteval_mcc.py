from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from transformers import AutoTokenizer
from lightning import Trainer
from pathlib import Path

from src.cls_ft import compute_2d_mcc_loss, make_loader, CLSFineTuner
from src.model import cls_load_tweeteval
from src.utils import CustomWriter, get_plt_loggers, MonitoringMetrics


EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
PATIENCE = 5
MAX_LEN = 125
MODEL_NAME = "cardiffnlp/twitter-roberta-large-2022-154m"

RESULT_PATH = Path('results/cardiff_tweeteval_mcc')

if not RESULT_PATH.is_dir():
    RESULT_PATH.mkdir()

def run(k):

    result_path =  RESULT_PATH / str(k)
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

    model = CLSFineTuner(
        MODEL_NAME, 
        compute_2d_mcc_loss, 
        learning_rate=LEARNING_RATE
    )
    pred_writer = CustomWriter(output_dir=result_path, write_interval="epoch")

    trainer = Trainer(
        default_root_dir=result_path,
        max_epochs=EPOCHS, 
        log_every_n_steps=1, 
        logger=get_plt_loggers(result_path, MODEL_NAME.split('/')[-1]),
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"), pred_writer],
        accelerator="gpu", devices=8, strategy="deepspeed_stage_2", precision=16,
    )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model.model.save_pretrained(result_path)

    monitor.set_time('training')

    trainer.predict(model, test_dataloader, return_predictions=False)

    monitor.set_time('predicting')

    monitor.save(result_path / 'monitoring.json')

if __name__ == '__main__':
    import sys
    k = sys.argv[1]
    run(k)