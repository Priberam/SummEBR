from nltk.probability import log_likelihood
import torch
import torch.nn.functional as F
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
        factual_reg: float = 0.0,
        factual_loss_margin: float = 0.2,
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

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, encoder_outputs=None):
        return self.bart(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels,
                         encoder_outputs=encoder_outputs)

    def forward_and_loss(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        decoder_input_ids = batch['decoder_input_ids']
        claim_tok_pos = batch['claim_tok_pos0']
        claim_tok_neg = batch['claim_tok_neg0']

        outputs = self(input_ids=input_ids, attention_mask=attention_mask,
                       decoder_input_ids=decoder_input_ids, labels=labels)
        ce_loss = outputs.loss

        if self.hparams.factual_reg > 0:
            encoder_outputs = (outputs.encoder_last_hidden_state, outputs.encoder_hidden_states, outputs.encoder_attentions)

            outputs_pos = self(encoder_outputs=encoder_outputs,
                               decoder_input_ids=claim_tok_pos, labels=claim_tok_pos)
            logits_pos = outputs_pos.logits

            outputs_neg = self(encoder_outputs=encoder_outputs,
                               decoder_input_ids=claim_tok_neg, labels=claim_tok_neg)
            logits_neg = outputs_neg.logits

            fact_loss = self.factual_loss(logits_pos, logits_neg, claim_tok_pos, claim_tok_neg, margin=self.hparams.factual_loss_margin)
            loss = ce_loss + self.hparams.factual_reg * fact_loss
        else:
            fact_loss = None
            loss = ce_loss

        return loss, ce_loss, fact_loss

    def training_step(self, batch, batch_idx):
        loss, ce_loss, fact_loss = self.forward_and_loss(batch)

        if fact_loss is not None:
            self.log('train_ce_loss', ce_loss, on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
            self.log('train_factual_loss', fact_loss, on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, ce_loss, fact_loss = self.forward_and_loss(batch)

        if fact_loss is not None:
            self.log('val_ce_loss', ce_loss, on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
            self.log('val_factual_loss', fact_loss, on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

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
                self.show_examples(
                    input_ids.cpu(), labels.cpu(), preds=preds.cpu(), ofile=f)

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

    @staticmethod
    def factual_loss(logits_pos, logits_neg, labels_pos, labels_neg, margin=0.1):
        assert len(logits_pos) == len(logits_neg), 'Number of positive and negative examples must be the same.'

        N, Lpos, _ = logits_pos.shape
        Lneg = logits_neg.shape[1]
        len_pos = torch.sum(labels_pos != -100, dim=1)
        loss_pos = F.cross_entropy(logits_pos.reshape(
            N*Lpos, -1), labels_pos.reshape(N*Lpos), reduction='none').reshape(N, Lpos).sum(dim=1) / len_pos
        len_neg = torch.sum(labels_neg != -100, dim=1)
        loss_neg = F.cross_entropy(logits_neg.reshape(
            N*Lneg, -1), labels_neg.reshape(N*Lneg), reduction='none').reshape(N, Lneg).sum(dim=1) / len_neg
        return torch.max(torch.zeros_like(loss_pos), loss_pos - loss_neg + margin).mean()

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
