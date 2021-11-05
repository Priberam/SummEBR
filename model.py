import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule
from datasets import load_metric
import nltk
import numpy as np
import sys

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
        ofile: str = 'output.txt',
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

        self.metric = load_metric('rouge')

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        return self.bart(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        decoder_input_ids = batch['decoder_input_ids']

        outputs = self(input_ids, attention_mask,
                       decoder_input_ids=decoder_input_ids, labels=labels)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        decoder_input_ids = batch['decoder_input_ids']

        outputs = self(input_ids, attention_mask,
                       decoder_input_ids=decoder_input_ids, labels=labels)
        val_loss, logits = outputs[:2]
        self.log('val_loss', val_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        preds = self.bart.generate(
            input_ids, attention_mask=attention_mask, max_length=1024)

        outputs = self.compute_metrics(
            (preds.cpu().numpy(), labels.cpu().numpy()))
        outputs['batch_size'] = len(input_ids)

        if batch_idx == 0:
            with open(self.hparams.ofile, 'w') as f:
                self.show_examples(input_ids.cpu(), labels.cpu(), preds=preds.cpu(), ofile=f)

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

    metric = load_metric("rouge")

    @staticmethod
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = [np.where(label != -100, label,
                           self.tokenizer.pad_token_id) for label in labels]
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(
            decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds,
                                     references=decoded_labels,
                                     use_stemmer=True, use_agregator=False)

        # Extract a few results from ROUGE
        result = {key: sum(x.fmeasure * 100 for x in lst)/len(lst)
                  for key, lst in result.items()}

        prediction_lens = [np.count_nonzero(
            pred != self.tokenizer.pad_token_id) for pred in preds]
        result['gen_len'] = np.mean(prediction_lens)
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
