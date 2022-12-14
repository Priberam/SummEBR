import argparse
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from dataset import RankDataMod
from models import BertRanker


def main():
    parser = argparse.ArgumentParser(
        description='BERT-based summary ranker.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-d', '--data_path', default='./data/cnndm/rerank_data',
                        type=str, metavar='', help='data directory path')
    parser.add_argument('-o', '--output', default='./checkpoints/cnndm',
                        type=str, metavar='', help='checkpoint save dir')
    parser.add_argument('--model_name_or_path', default='bert-base-uncased',
                        type=str, metavar='', help='path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--tokenizer_name', default=None,
                        type=str, metavar='', help='path to tokenizer if not the same as model_name')
    parser.add_argument('--config_name', default=None,
                        type=str, metavar='', help='path to config if not the same as model_name')
    parser.add_argument('--loss', default='listmle',
                        type=str, metavar='', help='training loss function')
    parser.add_argument('--temperature', default=1.,
                        type=float, metavar='', help='temperature hyperparameter for softmax')
    parser.add_argument('--margin_weight', default=10.,
                        type=float, metavar='', help='margin weight hyperparameter for max_margin loss')
    parser.add_argument('--metric', default='ctc_sum',
                        type=str, metavar='', help='ranking metric')
    parser.add_argument('--learning_rate', default=5e-5,
                        type=float, metavar='', help='learning rate')
    parser.add_argument('--batch_size', default=4,
                        type=int, metavar='', help='batch size')
    parser.add_argument('--num_train_samples', default=-1,
                        type=int, metavar='', help='number of training examples')
    parser.add_argument('--checkpoint', default=None,
                        type=str, metavar='', help='checkpoint file')
    parser.add_argument('--logdir', default=None,
                        type=str, metavar='', help='logs save directory')
    parser.add_argument('--predictions_file', default='./predictions.jsonl',
                        type=str, metavar='', help='output predictions file (.jsonl)')
    parser.add_argument('--do_train', dest='do_train', action='store_true')
    parser.add_argument('--do_eval', dest='do_eval', action='store_true')
    parser.add_argument('--do_predict', dest='do_predict', action='store_true')
    parser.add_argument('--seed', default=33,
                        type=int, metavar='', help='random seed')
    args = parser.parse_args()
    if args.gpus is None:
        args.gpus = 0

    if args.seed > 0:
        seed_everything(args.seed)

    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    model = BertRanker(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        config_name=args.config_name,
        loss=args.loss,
        temperature=args.temperature,
        margin_weight=args.margin_weight,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        metric=args.metric,
        num_train_samples=args.num_train_samples,
        predictions_file=args.predictions_file,
    )

    datamodule = RankDataMod(
        path=args.data_path,
        metric=args.metric,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        predict_only=not args.do_train
    )

    checkpoint = args.checkpoint
    if args.do_train:
        if checkpoint is not None:
            model = BertRanker.load_from_checkpoint(checkpoint)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_ndcg',
            dirpath=args.output,
            filename=f'ebr-{args.metric}-{args.loss}-'+'{epoch:02d}-{val_ndcg:.2f}',
            save_top_k=3,
            mode='max',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = TensorBoardLogger(save_dir=os.getcwd(), name=args.logdir)

        trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor], logger=logger)
        trainer.validate(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
        checkpoint = checkpoint_callback.best_model_path

    if args.do_eval:
        if checkpoint is not None:
            model = BertRanker.load_from_checkpoint(checkpoint)
        trainer = Trainer.from_argparse_args(args, logger=False)
        trainer.test(model, datamodule=datamodule)

    if args.do_predict:
        if checkpoint is not None:
            model = BertRanker.load_from_checkpoint(checkpoint)
            model.predictions_file = args.predictions_file
        trainer = Trainer.from_argparse_args(args, logger=False)
        trainer.predict(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
