import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule
from datasets import load_metric
from ctc_score import SummarizationScorer
from questeval.questeval_metric import QuestEval
from factsumm import FactSumm
from sklearn import metrics
import nltk
import numpy as np
import sys
import json
from tqdm import tqdm
from utils import postprocess_text, replace_special_chars

class BartSummarizer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: AutoTokenizer = None,
        config_name: str = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
        num_return_sequences: int = 1,
        predictions_file: str = 'predictions.jsonl',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['predictions_file'])

        self.predictions_file = predictions_file
        self.tokenizer = tokenizer
        config_name = config_name if config_name is not None else model_name_or_path
        config = AutoConfig.from_pretrained(
            config_name,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
        )
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
        )

        self.rouge = load_metric('rouge')
        self.ctc_scorer = SummarizationScorer(align='D-cnndm')
        self.questeval_scorer = QuestEval(task='summarization', do_weighter=True)
        self.factsumm_scorer = FactSumm()

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        return self.bart(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = {}
        outputs['val_loss'] = self(**batch).loss

        preds = self.bart.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=128).cpu()
        outputs.update(
            self.compute_metrics(
                batch['input_ids'].cpu().numpy(),
                preds.numpy(),
                batch['labels'].cpu().numpy(),
                metrics=['rouge', 'questeval']
            )
        )
        outputs['batch_size'] = len(batch['input_ids'])

        for key in outputs:
            if key != 'batch_size':
                self.log(key, outputs[key], on_step=True, on_epoch=True, prog_bar=True)

        return outputs

    def validation_epoch_end(self, outputs):
        metrics = {}
        num_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / num_examples
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        preds = self.bart.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=128).cpu()
        input_ids = batch['input_ids'].cpu()
        labels = batch['labels'].cpu()

        outputs = self.compute_metrics(
            input_ids.numpy(), preds.numpy(), labels.numpy(),
            metrics=['rouge', 'questeval', 'ctc']
        )
        outputs['batch_size'] = len(input_ids)
        return outputs

    def test_epoch_end(self, outputs):
        metrics = {}
        num_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / num_examples
        self.log_dict(metrics)
        return metrics

    def on_predict_start(self):
        self._predict_f = open(self.predictions_file, 'w', encoding='utf-8')

    def predict_step(self, batch, batch_idx):
        preds = self.bart.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_beams=self.hparams.num_beams,
            num_beam_groups=self.hparams.num_beam_groups,
            diversity_penalty=self.hparams.diversity_penalty,
            num_return_sequences=self.hparams.num_return_sequences,
        ).cpu().numpy()
        input_ids = batch['input_ids'].cpu().numpy()
        labels = [np.where(label != -100, label, self.tokenizer.pad_token_id)
                  for label in batch['labels'].cpu().numpy()]

        decoded_inputs = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        if self.hparams.num_return_sequences > 1:
            decoded_preds = [decoded_preds[j : j+self.hparams.num_return_sequences]
                             for j in range(0, len(decoded_preds), self.hparams.num_return_sequences)]

        for inpt, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
            example = {'text': inpt, 'gold_summary': label}
            if self.hparams.num_return_sequences == 1:
                example['gen_summary'] = pred
            else:
                pred = list(set(pred))
                for j in range(len(pred)):
                    example[f'gen_summary{j}'] = pred[j]
            self._predict_f.write(json.dumps(example, ensure_ascii=False) + '\n')

    def on_predict_end(self):
        self._predict_f.close()

    def setup(self, stage=None) -> None:
        if stage != 'fit':
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        steps_per_epoch = (len(train_loader.dataset) //
                           tb_size) // self.trainer.accumulate_grad_batches
        self.total_steps = self.trainer.max_epochs * steps_per_epoch

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.bart
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.10 * self.total_steps),
            num_training_steps=self.total_steps,
        )
        scheduler = {'scheduler': scheduler,
                     'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @staticmethod
    def postprocess_text(inputs, preds, labels):
        inputs = [inpt.strip() for inpt in inputs]
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        inputs = ["\n".join(nltk.sent_tokenize(inpt)) for inpt in inputs]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return inputs, preds, labels

    def compute_metrics(self, input_ids, preds, labels, metrics=None, input_is_text=False):
        result = {}

        if not input_is_text:
            decoded_inputs = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True)
            decoded_preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = [np.where(label != -100, label,
                            self.tokenizer.pad_token_id) for label in labels]
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            prediction_lens = [np.count_nonzero(
                pred != self.tokenizer.pad_token_id) for pred in preds]
            result['gen_len'] = np.mean(prediction_lens)
        else:
            decoded_inputs, decoded_preds, decoded_labels = input_ids, preds, labels
            preds_tok = self.tokenizer(decoded_preds)['input_ids']
            prediction_lens = [len(pred) for pred in preds_tok]
            result['gen_len'] = np.mean(prediction_lens)

        # Some simple post-processing
        decoded_inputs = postprocess_text(decoded_inputs)
        decoded_preds = postprocess_text(decoded_preds)
        decoded_labels = postprocess_text(decoded_labels)

        if metrics is None or 'rouge' in metrics:
            # Extract a few results from ROUGE
            rouge_scores = self.rouge.compute(predictions=decoded_preds,
                                              references=decoded_labels,
                                              use_stemmer=True, use_agregator=False)

            result.update({key: sum(x.fmeasure * 100 for x in lst)/len(lst)
                           for key, lst in rouge_scores.items()})

        if metrics is None or 'ctc' in metrics:
            consistency_scores, relevance_scores = [], []
            for inpt, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                inpt = replace_special_chars(inpt)
                pred = replace_special_chars(pred)
                label = replace_special_chars(label)

                try:
                    consistency = self.ctc_scorer.score(doc=inpt, refs=[], hypo=pred, aspect='consistency')
                    relevance = self.ctc_scorer.score(doc=inpt, refs=[label], hypo=pred, aspect='relevance')
                    consistency_scores.append(consistency if consistency is not None else 0)
                    relevance_scores.append(relevance if relevance is not None else 0)
                except:
                    print('Couldn\'t compute CTC scores for the current example. Skipping it.')
            result['ctc_consistency'] = np.mean(consistency_scores)
            result['ctc_relevance'] = np.mean(relevance_scores)

        if metrics is None or 'questeval' in metrics:
            result['questeval'] = self.questeval_scorer.corpus_questeval(
                hypothesis=decoded_preds,
                sources=decoded_inputs)['corpus_score']

        if metrics is None or 'factsumm' in metrics:
            rouge, open_fact, closed_fact, qags = [], [], [], []
            for inpt, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                rouge.append(self.factsumm_scorer.calculate_rouge(label, pred))
                open_fact.append(self.factsumm_scorer.extract_triples(inpt, pred, verbose=False))
                closed_fact.append(self.factsumm_scorer.extract_facts(inpt, pred, device='cuda', verbose=False))
                qags.append(self.factsumm_scorer.extract_qas(inpt, pred, device='cuda', verbose=False))

            result['factsumm_rouge1'] = np.mean([x[0] for x in rouge])
            result['factsumm_rouge2'] = np.mean([x[1] for x in rouge])
            result['factsumm_rougeL'] = np.mean([x[2] for x in rouge])
            result['factsumm_openfact'] = np.mean(open_fact)
            result['factsumm_closedfact'] = np.mean([x[2] for x in closed_fact])
            result['factsumm_qags'] = np.mean(qags)

        return result

    def show_examples(self, input_ids, labels, preds=None, ofile=None):

        decoded_inputs = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True)

        labels = [np.where(label != -100, label,
                           self.tokenizer.pad_token_id) for label in labels]
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        if preds is not None:
            decoded_preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True)
        else:
            decoded_preds = [None] * len(decoded_inputs)

        ofile = ofile or sys.stdout
        for i, (input, label, pred) in enumerate(zip(decoded_inputs, decoded_labels, decoded_preds)):
            print(f'Article {i}:', file=ofile)
            print(input, file=ofile)
            print(file=ofile)

            print(f'Ref. summary {i}:', file=ofile)
            print(label, file=ofile)
            print(file=ofile)

            if pred is not None:
                print(f'Gen. summary {i}:', file=ofile)
                print(pred, file=ofile)
                print(file=ofile)

    def testfromjson(self, filename, batch_size):
        with open(filename, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        examples = [json.loads(line) for line in tqdm(lines)]

        outputs, input_batch, pred_batch, label_batch = [], [], [], []
        for example in tqdm(examples):
            input_batch.append(example['text'])
            pred_batch.append(example['gen_summary'])
            label_batch.append(example['gold_summary'])

            if len(input_batch) == batch_size:
                output = self.compute_metrics(input_batch, pred_batch, label_batch, metrics=['rouge', 'questeval', 'ctc'], input_is_text=True)
                output['batch_size'] = batch_size
                outputs.append(output)
                input_batch, pred_batch, label_batch = [], [], []
        if(input_batch):
            output = self.compute_metrics(input_batch, pred_batch, label_batch, metrics=['rouge', 'questeval', 'ctc'], input_is_text=True)
            output['batch_size'] = len(input_batch)
            outputs.append(output)

        metrics = {}
        num_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / num_examples
        return metrics


class BertRanker(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: AutoTokenizer = None,
        config_name: str = None,
        loss: str = 'listmle',
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        metric: str = 'ctc_sum',
        num_train_samples: int =-1,
        predictions_file: str = 'predictions.jsonl'
    ):
        assert loss in ['listmle', 'nce']

        super().__init__()
        self.save_hyperparameters(ignore=['predictions_file'])

        self.tokenizer = tokenizer
        config_name = config_name if config_name is not None else model_name_or_path
        config = AutoConfig.from_pretrained(config_name)
        config.num_labels = 1
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )
        self.loss_fn = self.listmle_loss if loss == 'listmle' else self.nce_loss
        self.predictions_file = predictions_file

    def forward(
        self,
        batch: dict,
        infty: float = 1e9,
    ):
        N = len(batch['num_candidates'])
        energies, i = [], 0
        while True:
            if f'cand{i}_ids' not in batch:
                break

            outputs = self.bert(
                input_ids=batch[f'cand{i}_ids'],
                attention_mask=batch[f'cand{i}_attention_mask'],
                token_type_ids=batch[f'cand{i}_type_ids'],
            )
            energies.append(outputs.logits.reshape(N))
            i += 1
        energies = torch.stack(energies, dim=1)

        N, L = energies.shape
        mask = torch.arange(L).unsqueeze(0).repeat(N, 1).to(energies.device)
        mask -= batch['num_candidates'].unsqueeze(1)
        mask = (mask < 0)
        energies[~mask] = infty
        return energies, mask

    @staticmethod
    def listmle_loss(scores, mask):
        loss = 0.
        max_num_candidates = mask.sum(dim=1).max().item()
        for i in range(max_num_candidates):
            scores_i = scores[:, i:].clone()
            scores_i -= scores_i.max(dim=1, keepdim=True).values
            top_score = scores_i[:, 0]
            loss_per_example = scores_i.logsumexp(dim=1) - top_score
            loss_per_example[~mask[:, i]] = 0
            num_valid = mask[:, i].long().sum()
            loss += loss_per_example.sum() / num_valid
        loss /= scores.shape[1]
        return loss

    @staticmethod
    def nce_loss(scores, mask):
        N = scores.shape[0]
        with torch.no_grad():
            probs = scores.exp()  # no need to normalize for torch.multinomial
            probs[~mask] = 0
        indices = torch.multinomial(probs, num_samples=2)
        indices = indices.sort(dim=1).values
        scores_pos = scores[torch.arange(N), indices[:, 0]]
        scores_neg = scores[torch.arange(N), indices[:, 1]]
        logits = torch.cat([scores_pos, scores_neg], dim=0)
        target = torch.cat([
            torch.ones_like(scores_pos),
            torch.zeros_like(scores_neg),
        ], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss

    @staticmethod
    def ndcg_metric(scores, mask):
        N, L = scores.shape
        true_relevances = 2**torch.arange(L-1, -1, step=-1).unsqueeze(0).repeat(N, 1) - 1
        true_relevances[~mask] = 0
        ndcg = metrics.ndcg_score(true_relevances.cpu().numpy(), scores.cpu().numpy(), ignore_ties=True)
        return ndcg

    def training_step(self, batch, batch_idx):
        energies, mask = self(batch)
        loss = self.loss_fn(-energies, mask)
        self.log('train_loss', loss)

        with torch.no_grad():
            preds = energies.argmin(dim=1)
        top1acc = (preds == 0).float().mean()
        top3acc = (preds < 3).float().mean()
        self.log('train_top1_acc', top1acc)
        self.log('train_top3_acc', top3acc)

        return loss

    def validation_step(self, batch, batch_idx):
        energies, mask = self(batch)
        outputs = {}
        outputs['val_loss'] = self.loss_fn(-energies, mask)

        preds = energies.argmin(dim=1)
        outputs['val_top1_acc'] = (preds == 0).float().mean()
        outputs['val_top3_acc'] = (preds < 3).float().mean()

        outputs['val_ndcg'] = self.ndcg_metric(-energies, mask)

        for key in outputs:
            self.log(key, outputs[key])

        outputs['batch_size'] = len(energies)

        return outputs

    def validation_epoch_end(self, outputs):
        metrics = {}
        num_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / num_examples
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        energies, mask = self(batch)
        outputs = {}
        outputs['test_loss'] = self.loss_fn(-energies, mask)

        preds = energies.argmin(dim=1)
        outputs['test_top1_acc'] = (preds == 0).float().mean()
        outputs['test_top3_acc'] = (preds < 3).float().mean()

        outputs['test_ndcg'] = self.ndcg_metric(-energies, mask)

        outputs['batch_size'] = len(energies)

        return outputs

    def test_epoch_end(self, outputs):
        metrics = {}
        num_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / num_examples
        self.log_dict(metrics)
        return metrics

    def on_predict_start(self):
        self._predict_f = open(self.predictions_file, 'w', encoding='utf-8')

    def predict_step(self, batch, batch_idx):
        energies, _ = self(batch)
        preds = energies.argmin(dim=1)
        for i, pred in enumerate(preds):
            token_ids = batch[f'cand{pred}_ids'][i]
            type_ids = batch[f'cand{pred}_type_ids'][i]
            source_ids = token_ids[type_ids == 0]
            summary_ids = token_ids[type_ids == 1]
            source_txt = self.tokenizer.decode(source_ids, skip_special_tokens=True)
            summary_txt = self.tokenizer.decode(summary_ids, skip_special_tokens=True)
            candidate_rank = pred.item()
            summary_idx = batch['candidate_indices'][i, candidate_rank].item()
            example = {
                'text': source_txt,
                'summary': summary_txt,
                'summary_index': summary_idx,
                'candidate_rank': candidate_rank
            }
            self._predict_f.write(json.dumps(example, ensure_ascii=False) + '\n')

    def on_predict_end(self):
        self._predict_f.close()

    def setup(self, stage=None) -> None:
        if stage != 'fit':
            return
        train_loader = self.train_dataloader()

        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        steps_per_epoch = (len(train_loader.dataset) //
                           tb_size) // self.trainer.accumulate_grad_batches
        self.total_steps = self.trainer.max_epochs * steps_per_epoch

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps,
        )
        scheduler = {'scheduler': scheduler,
                     'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
