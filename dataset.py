from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
from datasets import load_from_disk
from datasets import Dataset as HF_Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import json
import random
import os


class CnnDmDataMod(LightningDataModule):

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
        self.collate_fn = DataCollatorForCnnDmAug(
            tokenizer=tokenizer,
            model=model
        )

        self.dataset = {
            'train': CnnDmAug(path, 'train', tokenizer),
            'validation': CnnDmAug(path, 'validation', tokenizer),
            'test': CnnDmAug(path, 'test', tokenizer, n_pairs=0),  # no augmented data in the test set
        }

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

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


class CnnDmAug(Dataset):
    def __init__(self, path, split, tokenizer, max_seq_length=1024, n_pairs=1, source='both', cache=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.n_pairs = n_pairs

        print('Processing raw data...')
        self.data = load_from_disk(path)[split]
        # self.data = self.data.filter(
        #     lambda x, idx: True if idx < 3 else False, with_indices=True)
        self.data = self.data.map(
            self.convert_to_features,
            batched=True,
            batch_size=1000,
        )
        self.data.set_format(type="torch", columns=[
                            'input_ids', 'attention_mask', 'labels'])

        if self.n_pairs > 0:
            print('Processing augmented data...')
            self.read_and_process_claims(path, split, source, cache=cache)


    @staticmethod
    def read_jsonl(input_file):
        '''Read a .jsonl file into a pandas DataFrame'''
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
        df = pd.DataFrame(lines)
        return df

    def read_and_process_claims(self, path, split, source, cache=True):
        '''Read, tokenize, and pair positive and negative claims.'''
        assert source in ['both', 'highlights',
                          'article'], 'source must be either "both", "highlights", or "article"'

        loaded = False
        if cache:
            cache_path = os.path.join(path, 'cache_aug', split)
            if os.path.exists(cache_path):
                self.aug_data = load_from_disk(cache_path)
                self.aug_data = self.aug_data.to_pandas()
                self.aug_data = self.aug_data.set_index('id')
                loaded = True
                print(f'Loaded processed augmented data at f{cache_path}')

        if not loaded:
            data_pos = self.read_jsonl(os.path.join(
                path, f'claims-{split}-positive.jsonl'))
            data_pos = data_pos.drop(['text'], axis=1)
            data_neg = self.read_jsonl(os.path.join(
                path, f'claims-{split}-negative.jsonl'))
            data_neg = data_neg.drop(['text'], axis=1)

            data_pos = HF_Dataset.from_pandas(data_pos)
            if source != 'both':
                data_pos = data_pos.filter(lambda x: x['source'] == source)
            data_pos = data_pos.map(
                self.convert_claims_to_features,
                batched=True,
                batch_size=1000,
            )
            data_pos = data_pos.to_pandas()
            data_pos = data_pos.set_index('id')

            data_neg = HF_Dataset.from_pandas(data_neg)
            if source != 'both':
                data_neg = data_neg.filter(lambda x: x['source'] == source)
            data_neg = data_neg.map(
                self.convert_claims_to_features,
                batched=True,
                batch_size=1000,
            )
            data_neg = data_neg.to_pandas()
            data_neg = data_neg.set_index('id')

            self.aug_data = pd.merge(data_pos, data_neg, on=[
                                    'id', 'source', 'sentence_id'], suffixes=('_pos', '_neg'))

            if cache:
                aug_data_hf = HF_Dataset.from_pandas(self.aug_data)
                aug_data_hf.save_to_disk(cache_path)

        # self.aug_data = []
        # for i in tqdm(range(aug_data.index.max() + 1)):
        #     try:
        #         self.aug_data.append(HF_Dataset.from_pandas(aug_data.loc[[i]]))
        #     except:
        #         print('Key not found:', i)

        # aug_data = HF_Dataset.from_pandas(aug_data)
        # aug_data.set_format(type="torch", columns=[
        #     'id', 'claim_tok_pos', 'claim_tok_neg'])
        # max_id = max(aug_data['id'])
        # self.aug_data = [aug_data.filter(
        #     lambda x: x['id'] == i) for i in range(max_id + 1)]

    def convert_to_features(self, example_batch, indices=None):
        '''Tokenize the articles and summaries'''
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

    def convert_claims_to_features(self, example_batch, indices=None):
        '''Tokenize the claims'''
        with self.tokenizer.as_target_tokenizer():
            claims = self.tokenizer(
                example_batch['claim'], max_length=self.max_seq_length, padding=False, truncation=True
            )

        features = {}
        features['id'] = example_batch['id']
        features['source'] = example_batch['source']
        features['augmentation'] = example_batch['augmentation']
        features['claim_tok'] = [torch.tensor(x) for x in claims['input_ids']]
        features['claim_attention_mask'] = [
            torch.tensor(x) for x in claims['attention_mask']]

        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data[index]

        if self.n_pairs > 0:
            try:
                all_pairs = self.aug_data.loc[[index]]
                ridx = random.sample(range(len(all_pairs)),
                                     min(self.n_pairs, len(all_pairs)))
                pairs = all_pairs[ridx]
                for i in range(len(ridx)):
                    features[f'claim_tok_pos{i}'] = pairs['claim_tok_pos'][i]
                    features[f'claim_tok_neg{i}'] = pairs['claim_tok_neg'][i]
            except:
                for i in range(self.n_pairs):
                    features[f'claim_tok_pos{i}'] = torch.tensor([0])
                    features[f'claim_tok_neg{i}'] = torch.tensor([0])

        # if self.n_pairs > 0 and len(self.aug_data) > index:
        #     all_pairs = self.aug_data[index]
        #     ridx = random.sample(range(len(all_pairs)),
        #                          min(self.n_pairs, len(all_pairs)))
        #     pairs = all_pairs[ridx]
        #     for i in range(len(ridx)):
        #         features[f'claim_tok_pos{i}'] = pairs['claim_tok_pos'][i]
        #         features[f'claim_tok_neg{i}'] = pairs['claim_tok_neg'][i]

        return features


@dataclass
class DataCollatorForCnnDmAug(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batched_features = [{key: feature[key] for key in [
            'input_ids', 'attention_mask', 'labels']} for feature in features]
        batched_features = super().__call__(
            batched_features, return_tensors=return_tensors)

        i = 0
        for key in features[0].keys():
            if key in ['input_ids', 'attention_mask', 'labels']:
                continue
            claims = [feature[key] for feature in features]
            max_length = max(len(c) for c in claims)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * \
                    (max_length - len(feature[key]))
                if isinstance(feature[key], list):
                    feature[key] = torch.tensor(
                        feature[key] +
                        remainder if padding_side == "right" else remainder +
                        feature[key]
                    ).long()
                elif padding_side == "right":
                    feature[key] = torch.cat(
                        [feature[key], torch.tensor(remainder)]).long()
                else:
                    feature[key] = torch.cat(
                        [torch.tensor(remainder), feature[key]]).long()
            i += 1
            batched_features[key] = torch.stack(
                [feature[key] for feature in features])

        return batched_features


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/bart-large', use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')
    dm = CnnDmDataMod(
        '/mnt/hdd2/dpc/CNN_DailyMail_HuggingFace/3.0.0', model, tokenizer, batch_size=3)
    dl = dm.train_dataloader()
    x = next(iter(dl))
    # data = CnnDmAug(
    #     '/mnt/hdd2/dpc/CNN_DailyMail_HuggingFace/3.0.0', 'train', tokenizer)
