import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AutoTokenizer
from dataset import CnnDmDataMod
from model import BartSummarizer


def main():
    parser = argparse.ArgumentParser(
        description='Text summarization with BART.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-d', '--data_path', default='/mnt/hdd2/dpc/CNN_DailyMail_HuggingFace/3.0.0',
                        type=str, metavar='', help='data directory path')
    parser.add_argument('-o', '--output', default='./checkpoints',
                        type=str, metavar='', help='checkpoint save dir')
    parser.add_argument('--model_name_or_path', default='facebook/bart-large',
                        type=str, metavar='', help='path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--tokenizer_name', default=None,
                        type=str, metavar='', help='path to tokenizer if not the same as model_name')
    parser.add_argument('--config_name', default=None,
                        type=str, metavar='', help='path to config if not the same as model_name')
    parser.add_argument('--learning_rate', default=5e-5,
                        type=float, metavar='', help='learning rate')
    parser.add_argument('--batch_size', default=4,
                        type=int, metavar='', help='batch size')
    parser.add_argument('--checkpoint', default=None,
                        type=str, metavar='', help='checkpoint file')
    parser.add_argument('--eval_external_file', default=None,
                        type=str, metavar='', help='./predictions.jsonl')
    parser.add_argument('--num_beams', default=1,
                        type=int, metavar='', help='number of beams for beam search')
    parser.add_argument('--num_beam_groups', default=1,
                        type=int, metavar='', help='number of groups for diverse beam search')
    parser.add_argument('--num_return_sequences', default=1,
                        type=int, metavar='', help='number of returned sequences for each input sequence')
    parser.add_argument('--diversity_penalty', default=0.,
                        type=float, metavar='', help='diversity penalty for diverse beam search')
    parser.add_argument('--predictions_file', default='./predictions.jsonl',
                        type=str, metavar='', help='output predictions file (.jsonl)')
    parser.add_argument('--do_train', dest='do_train', action='store_true')
    parser.add_argument('--do_eval', dest='do_eval', action='store_true')
    parser.add_argument('--do_eval_external', dest='do_eval_external', action='store_true')
    parser.add_argument('--do_predict', dest='do_predict', action='store_true')
    parser.add_argument('--seed', default=33,
                        type=int, metavar='', help='random seed')
    args = parser.parse_args()

    if args.seed > 0:
        seed_everything(args.seed)

    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    model = BartSummarizer(
        args.model_name_or_path,
        tokenizer=tokenizer,
        config_name=args.config_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        num_return_sequences=args.num_return_sequences,
        predictions_file=args.predictions_file,
    )

    datamodule = CnnDmDataMod(
        args.data_path,
        model.bart,
        tokenizer,
        max_seq_length=1024,
        batch_size=args.batch_size,
    )

    checkpoint = args.checkpoint
    if args.do_train:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss_epoch',
            dirpath=args.output,
            filename="bart-summarizer-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor])
        trainer.fit(model, datamodule=datamodule)

        if args.do_eval:
            checkpoint = checkpoint_callback.best_model

    if args.do_eval:
        if checkpoint is not None:
            model = BartSummarizer.load_from_checkpoint(checkpoint)
        trainer = Trainer.from_argparse_args(args)
        trainer.test(model, datamodule=datamodule)

    if args.do_predict:
        if checkpoint is not None:
            model = BartSummarizer.load_from_checkpoint(checkpoint)
        trainer = Trainer.from_argparse_args(args)
        trainer.predict(model, dataloaders=datamodule.test_dataloader())

    if args.do_eval_external:
        metrics = model.testfromjson(args.eval_external_file, args.batch_size)
        print(f'Results from input {args.eval_external_file}:')
        print(metrics)


if __name__ == '__main__':
    main()
