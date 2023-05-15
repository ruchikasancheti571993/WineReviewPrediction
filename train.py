import torch
import transformers
from torch.utils.data import Dataset
from transformers import (ElectraForSequenceClassification,
                          ElectraTokenizerFast, EvalPrediction, InputFeatures,
                          Trainer, TrainingArguments, glue_compute_metrics)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np

#Load model
model = ElectraForSequenceClassification.from_pretrained(
    "google/electra-small-discriminator", num_labels = 19)

tokenizer = ElectraTokenizerFast.from_pretrained(
    "google/electra-small-discriminator", do_lower_case=True)    

class TrainerDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer

        # Tokenize the input
        self.tokenized_inputs = tokenizer(inputs, padding=True,truncation=True)   

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return InputFeatures(
            input_ids=self.tokenized_inputs['input_ids'][idx],
            token_type_ids=self.tokenized_inputs['token_type_ids'][idx],
            attention_mask=self.tokenized_inputs['attention_mask'][idx],
            label=self.targets[idx])  
    
def tokenized_dataset(train,val,test):
    train_dataset = TrainerDataset(train["description"],
                                train["taster_label"], tokenizer)

    val_dataset = TrainerDataset(val["description"],
                                val["taster_label"], tokenizer)

    test_dataset = TrainerDataset(test["description"],
                                test["taster_label"], tokenizer)
    
    return train_dataset,val_dataset,test_dataset
    
def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    # The choice of a dataset (task_name) implies metric
    return glue_compute_metrics(
        task_name="sst-2",
        preds=preds,
        labels=p.label_ids)

def start_training(train_dataset,val_dataset,test_dataset):
# Set seed for reproducibility
    np.random.seed(123)
    torch.manual_seed(123)

    training_args = TrainingArguments(
        output_dir="./models/model_electra",
        num_train_epochs=3,  # 1 (1 epoch gives slightly lower accuracy)
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,  
        dataloader_drop_last=True,  # Make sure all batches are of equal size
        load_best_model_at_end=True,
        evaluation_strategy="steps"
    )
        
    # Instantiate the Trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics)
    
    trainer.train()
    return trainer


    