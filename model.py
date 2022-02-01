import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule
from datasets import load_metric
from ctc_score import SummarizationScorer
from questeval.questeval_metric import QuestEval
from factsumm import FactSumm
import nltk
import numpy as np
import sys
import json
from tqdm import tqdm

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
        predictions_file: str = 'predict.jsonl',
    ):
        super().__init__()
        self.save_hyperparameters()

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
        n_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / n_examples
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
        n_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / n_examples
        self.log_dict(metrics)
        return metrics

    def on_predict_start(self):
        self._predict_f = open(self.hparams.predictions_file, 'w', encoding='utf-8')

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
        decoded_inputs, decoded_preds, decoded_labels = self.postprocess_text(
            decoded_inputs, decoded_preds, decoded_labels)

        if metrics is None or 'rouge' in metrics:
            # Extract a few results from ROUGE
            rouge_scores = self.rouge.compute(predictions=decoded_preds,
                                              references=decoded_labels,
                                              use_stemmer=True, use_agregator=False)

            result.update({key: sum(x.fmeasure * 100 for x in lst)/len(lst)
                           for key, lst in rouge_scores.items()})

            rouge_source = self.rouge.compute(predictions=decoded_preds,
                                              references=decoded_labels,
                                              use_stemmer=True, use_agregator=False)
            result.update(
                {key+'_prec_src': sum(x.precision * 100  for x in lst)/len(lst)
                 for key, lst in rouge_source.items()}
            )

        if metrics is None or 'ctc' in metrics:
            consistency_scores, relevance_scores = [], []
            for inpt, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                inpt = self.replace_special_chars(inpt)
                pred = self.replace_special_chars(pred)
                label = self.replace_special_chars(label)

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

    @staticmethod
    def replace_special_chars(text):
        text = text.replace('â‚¬', '€')
        text = text.replace('â', 'a')
        text = text.replace('Â', 'A')
        text = text.replace('å', 'a')
        text = text.replace('Å', 'A')
        text = text.replace('ă', 'a')
        text = text.replace('Ă', 'A')
        text = text.replace('ä', 'a')
        text = text.replace('Ä', 'A')
        text = text.replace('č', 'c')
        text = text.replace('Č', 'C')
        text = text.replace('ď', 'd')
        text = text.replace('Ď', 'D')
        text = text.replace('ě', 'e')
        text = text.replace('Ě', 'E')
        text = text.replace('ę', 'e')
        text = text.replace('Ę', 'E')
        text = text.replace('ê', 'e')
        text = text.replace('Ê', 'E')
        text = text.replace('ễ', 'e')
        text = text.replace('Ễ', 'E')
        text = text.replace('ǧ', 'g')
        text = text.replace('Ǧ', 'G')
        text = text.replace('ị', 'i')
        text = text.replace('Ị', 'I')
        text = text.replace('ľ', 'l')
        text = text.replace('Ľ', 'L')
        text = text.replace('ň', 'n')
        text = text.replace('Ň', 'N')
        text = text.replace('ö', 'ö')
        text = text.replace('Ö', 'O')
        text = text.replace('ř', 'r')
        text = text.replace('Ř', 'R')
        text = text.replace('š', 's')
        text = text.replace('Š', 'S')
        text = text.replace('ś', 's')
        text = text.replace('Ś', 'S')
        text = text.replace('ť', 't')
        text = text.replace('Ť', 'T')
        text = text.replace('ü', 'u')
        text = text.replace('Ü', 'U')
        text = text.replace('ủ', 'u')
        text = text.replace('Ủ', 'U')
        text = text.replace('ů', 'u')
        text = text.replace('Ů', 'U')
        text = text.replace('ý', 'y')
        text = text.replace('Ý', 'Y')
        text = text.replace('ŷ', 'y')
        text = text.replace('Ŷ', 'Y')
        text = text.replace('ž', 'z')
        text = text.replace('Ž', 'Z')
        text = text.replace('¥', 'Y')
        text = text.replace('½', '1/2')
        text = text.replace('¼', '1/4')
        text = text.replace('¾', '3/4')
        text = text.replace('¹', '1')
        text = text.replace('²', '2')
        text = text.replace('³', '3')
        text = text.replace('⁴', '4')
        text = text.replace('⁄', '/')
        text = text.replace('˚', 'deg')
        return text

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
        n_examples = sum(x['batch_size'] for x in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(x[key] * x['batch_size']
                               for x in outputs) / n_examples
        return metrics


class BertRanker(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: AutoTokenizer = None,
        config_name: str = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        config_name = config_name if config_name is not None else model_name_or_path
        config = AutoConfig.from_pretrained(config_name)
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            num_labels=1,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        return self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

