from transformers import TrainingArguments, default_data_collator
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from datasets import Dataset
import torch

from src.training import IronyTrainer, MCC_Loss
from src.utils import load_config

config = load_config()
config = config | {
    'OUTPUT_DIR':"results/mcc_test", 
    'RESULT_PATH':"results/mcc_test.jsonl",
    'LOSS_FUNCS': [
        (MCC_Loss(), 1), 
        # (torch.nn.BCELoss(), 1),
    ],
}

class MCCTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff1 = torch.nn.Sequential(
            torch.nn.Linear(30, 15),
            torch.nn.ReLU()
        ) 
        self.ff2 = torch.nn.Sequential(
            torch.nn.Linear(15, 2),
        )

    def forward(self, inputs, labels=[]):
        return {"logits":self.ff2(self.ff1(inputs))}
    
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=42)
train_set = Dataset.from_list([{'inputs':x, 'label':y} for x, y in zip(X_train, y_train)])
val_set = Dataset.from_list([{'inputs':x, 'label':y} for x, y in zip(X_val, y_val)])
test_set = Dataset.from_list([{'inputs':x, 'label':y} for x, y in zip(X_val, y_val)])

model = MCCTestModel()

training_args = TrainingArguments(
    output_dir=config['OUTPUT_DIR'],
    do_train =True,
    do_eval=True,
    evaluation_strategy='epoch',
    prediction_loss_only=False,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    num_train_epochs=50,
    save_strategy='epoch',
    save_total_limit=5,
    optim='adamw_torch',
    load_best_model_at_end=True,
    logging_strategy="epoch",
    fp16=True 
)

trainer = IronyTrainer(
    loss_funcs = config['LOSS_FUNCS'],
    model=model,
    args=training_args,
    train_dataset=train_set if training_args.do_train else None,
    eval_dataset=val_set if training_args.do_eval else None,
    data_collator=default_data_collator

)

trainer.train()
trainer.save_state()
