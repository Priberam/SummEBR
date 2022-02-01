from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import torch
from datasets import load_from_disk
from datasets import Dataset as HF_Dataset
import transformers
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import numpy as np
import os
from tqdm import tqdm
import json
transformers.logging.set_verbosity_error()
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
        self.collate_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )

        self.dataset = load_from_disk(self.path)
        # self.dataset = self.dataset.filter(
        #     lambda x, idx: True if idx < 32 else False, with_indices=True)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                batch_size=1000,
                remove_columns=['id'],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(
                type="torch", columns=['input_ids', 'attention_mask', 'labels'])

        self.eval_splits = [
            x for x in self.dataset.keys() if 'validation' in x]

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


class CnnDmRankDataset(Dataset):
    def __init__(self, path, split, metric, tokenizer, cache=True):
        super().__init__()
        assert split in ['train', 'validation', 'test']
        assert metric in ['ctc_relevance', 'ctc_consistency', 'ctc_sum', 'questeval', 'rougeL']

        if cache:
            try:
                print('Looking for cached preprocessed dataset...')
                self.tok_data = load_from_disk(os.path.join(path, 'cache', f'cache-{metric}-{split}.hf'))
                self.tok_data.set_format(type='torch')
            except:
                print("Couldn't find cached preprocessed dataset. Loading from raw data...")
                self.load_and_preprocess_data(path, split, metric, tokenizer)
        else:
            self.load_and_preprocess_data(path, split, metric, tokenizer)

        if cache:
            if not os.path.exists(os.path.join(path, 'cache')):
                os.makedirs(os.path.join(path, 'cache'))
            if not os.path.exists(os.path.join(path, 'cache', f'cache-{metric}-{split}.hf')):
                self.tok_data.save_to_disk(os.path.join(path, 'cache', f'cache-{metric}-{split}.hf'))

    def load_and_preprocess_data(self, path, split, metric, tokenizer):
        data_file = os.path.join(path, f'diverse-samples-{split}.jsonl')
        if 'ctc' in metric:
            rank_file = os.path.join(path, f'results-ctc-{split}.jsonl')
        else:
            rank_file = os.path.join(path, f'results-{metric.lower()}-{split}.jsonl')

        with open(data_file, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        examples = [json.loads(line) for line in tqdm(lines)]

        with open(rank_file, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        examples_rank = [json.loads(line) for line in tqdm(lines)]

        num_examples = min(len(examples), len(examples_rank)-1)
        examples = examples[:num_examples]
        examples_rank = examples_rank[:num_examples]

        if metric == 'ctc_relevance':
            rank_col = 'rank_relevance'
            metric_col = 'relevance'
        elif metric == 'ctc_consistency':
            rank_col = 'rank_consistency'
            metric_col = 'consistency'
        elif metric == 'ctc_sum':
            rank_col = 'rank_sum'
            metric_col = 'relevance'
        else:
            rank_col = 'rank'
            metric_col = metric

        text_data = []
        max_num_candidates = 0
        for i, (example, example_rank) in tqdm(enumerate(zip(examples, examples_rank)), total=num_examples):
            # if the current example has no valid score or has a single candidate, skip it
            if not np.any(example_rank[metric_col]) or len(example_rank[rank_col]) == 1:
                continue

            ranked_example = [example['text']]
            for rank in example_rank[rank_col]:
                ranked_example.append(example[f'gen_summary{rank}'])
            text_data.append(ranked_example)

            if len(example_rank[rank_col]) > max_num_candidates:
                max_num_candidates = len(example_rank[rank_col])

        tok_data = {'num_candidates': []}
        tok_data.update({f'cand{i}_ids': [] for i in range(max_num_candidates)})
        tok_data.update({f'cand{i}_type_ids': [] for i in range(max_num_candidates)})
        for i, example in tqdm(enumerate(text_data), total=len(text_data)):
            tok_data['num_candidates'].append(len(example)-1)
            for j in range(1, len(example)):
                example_tok = tokenizer(text=example[0], text_pair=example[j], truncation=True, return_overflowing_tokens=False)
                tok_data[f'cand{j-1}_ids'].append(example_tok['input_ids'])
                tok_data[f'cand{j-1}_type_ids'].append(example_tok['token_type_ids'])
            for j in range(len(example)-1, max_num_candidates):
                tok_data[f'cand{j}_ids'].append([tokenizer.pad_token_id])
                tok_data[f'cand{j}_type_ids'].append([tokenizer.pad_token_id])

        self.tok_data = HF_Dataset.from_dict(tok_data)
        self.tok_data.set_format(
            type='torch',
            columns=(['num_candidates']
                     + [f'cand{i}_ids' for i in range(max_num_candidates)]
                     + [f'cand{i}_type_ids' for i in range(max_num_candidates)]
                    ),
        )

    def __len__(self):
        return len(self.tok_data)

    def __getitem__(self, index):
        return self.tok_data[index]

class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batched_examples = {}
        processed_keys = []
        keys = list(examples[0].keys())
        for key in keys:
            if 'ids' not in key:
                batched_examples[key] = torch.stack(
                    [example[key] for example in examples])
                continue

            key_prefix = key.split('_')[0]
            if key_prefix in processed_keys:
                continue
            else:
                processed_keys.append(key_prefix)

            max_length = max(len(x[key_prefix + '_ids']) for x in examples)
            for example in examples:
                remainder = [self.pad_token_id] * \
                    (max_length - len(example[key_prefix + '_ids']))

                example[key_prefix + '_attention_mask'] = torch.cat([torch.ones_like(example[key_prefix + '_ids']), torch.zeros(len(remainder))]).bool()
                example[key_prefix + '_ids'] = torch.cat([example[key_prefix + '_ids'], torch.tensor(remainder)]).long()
                example[key_prefix + '_type_ids'] = torch.cat([example[key_prefix + '_type_ids'], torch.zeros(len(remainder))]).long()

            batched_examples[key_prefix + '_attention_mask'] = torch.stack(
                [example[key_prefix + '_attention_mask'] for example in examples])
            batched_examples[key_prefix + '_ids'] = torch.stack(
                [example[key_prefix + '_ids'] for example in examples])
            batched_examples[key_prefix + '_type_ids'] = torch.stack(
                [example[key_prefix + '_type_ids'] for example in examples])

        return batched_examples


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased', use_fast=False)
    data = CnnDmRankDataset('/mnt/hdd2/dpc/CNN_DailyMail_HuggingFace/3.0.0/rerank_data', 'validation', 'ctc_relevance', tokenizer)
    loader = DataLoader(data, batch_size=4, collate_fn=DataCollator(tokenizer.pad_token_id))
    next(iter(loader))
