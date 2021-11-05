from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import numpy as np

class CNN_DailyMail_DataModule(LightningDataModule):

    loader_columns = [
        'article', 'highlights'
    ]

    def __init__(
        self,
        path: str,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 1024,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.path = path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.collate_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )

    def setup(self, stage: str):
        self.dataset = load_from_disk(self.path)

        print('Setup start...')
        for split in self.dataset.keys():
            # self.dataset[split] = self.dataset[split].filter(
            #     lambda x, idx: True if idx < 32 else False, with_indices=True)
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                batch_size=(1000 // self.batch_size) * self.batch_size,
                remove_columns=['id'],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(
                type="torch", columns=['input_ids', 'attention_mask', 'labels'])

        self.eval_splits = [
            x for x in self.dataset.keys() if 'validation' in x]
        print('Setup done!')

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['validation'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, shuffle=False)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, shuffle=False) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, shuffle=False)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, shuffle=False) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Tokenize the text
        inputs = self.tokenizer(
            example_batch['article'], max_length=self.max_seq_length, padding=False, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                example_batch['highlights'], max_length=self.max_seq_length, padding=False, truncation=True
            )
        features = {}
        features['input_ids'] = [torch.tensor(x) for x in inputs['input_ids']]
        features['attention_mask'] = [torch.tensor(
            x) for x in inputs['attention_mask']]
        features['labels'] = [torch.tensor(x) for x in labels['input_ids']]

        return features
