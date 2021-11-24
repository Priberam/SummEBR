"""
Script for generating synthetic data for FactCC training.

Script expects source documents in `jsonl` format with each source document
embedded in a separate json object.

Json objects are required to contain `id` and `text` keys.
"""

import argparse
import json
import os
import random
import spacy
from tqdm import tqdm

import augmentation_ops as ops

def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def save_data(args, data, name_suffix):
    output_file = os.path.splitext(args.data_file)[0] + "-" + name_suffix + ".jsonl"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            example["text"] = example["text"].text
            example["claim"] = example["claim"].text
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_data(args, name_suffix):
    input_file = os.path.splitext(args.data_file)[0] + "-" + name_suffix + ".jsonl"
    nlp = spacy.load("en")

    examples = []
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as fd:
            lines = fd.readlines()
        for line in tqdm(lines):
            example = json.loads(line)
            example["text"] = nlp(example["text"])
            example["claim"] = nlp(example["claim"])
            examples.append(example)
    return examples


def apply_transformation(data, operation, batch_size=None):
    new_data = []
    batch = []
    for i, example in enumerate(tqdm(data)):
        if batch_size is not None:
            batch.append(example)
        else:
            batch = example
        if batch_size is None or len(batch) == batch_size or i == len(data) - 1:
            try:
                new_example = operation.transform(batch)
                if new_example:
                    if isinstance(new_example, list):
                        new_data.extend(new_example)
                    else:
                        new_data.append(new_example)
            except Exception as e:
                print("Caught exception:", e)
            batch = []
    return new_data


def main(args):
    random.seed(42)

    # create or load positive examples
    data = []
    if args.load_intermediate:
        data = load_data(args, "clean")
        print("Loaded %s example pairs." % len(data))
    if not data:
        # load data
        source_docs = load_source_docs(args.data_file, to_dict=False)
        print("Loaded %d source documents." % len(source_docs))

        print("Creating data examples")
        sclaims_op = ops.SampleSentences(num_samples=2, min_sent_len=8)
        data = apply_transformation(source_docs, sclaims_op)
        print("Created %s example pairs." % len(data))

        if args.save_intermediate:
            save_data(args, data, "clean")

    # backtranslate
    data_btrans = []
    if not args.augmentations or "backtranslation" in args.augmentations:
        if args.load_intermediate:
            data_btrans = load_data(args, "btrans")
            print("Loaded %s backtranslation example pairs." %
                  len(data_btrans))

        if not data_btrans:
            print("Creating backtranslation examples")
            btrans_op = ops.Backtranslation(api=args.translation_api)
            data_btrans = apply_transformation(data, btrans_op, batch_size=64)
            print("Backtranslated %s example pairs." % len(data_btrans))

            if args.save_intermediate:
                save_data(args, data_btrans, "btrans")

    data_positive = data + data_btrans
    data, data_btrans = [], []
    save_data(args, data_positive, "positive")

    # create negative examples
    data_pronoun = []
    if not args.augmentations or "pronoun_swap" in args.augmentations:
        if args.load_intermediate:
            data_pronoun = load_data(args, "pronoun")
            # for example in data_pronoun:
            #     example['text'].text = ''
            print("Loaded %s pronoun swap example pairs." % len(data_btrans))

        if not data_pronoun:
            print("Creating pronoun swap examples")
            pronoun_op = ops.PronounSwap()
            data_pronoun = apply_transformation(data_positive, pronoun_op)
            # for example in data_pronoun:
            #     example['text'].text = ''
            print("PronounSwap %s example pairs." % len(data_pronoun))

            if args.save_intermediate:
                save_data(args, data_pronoun, "pronoun")
    # data_pronoun = []

    data_dateswp = []
    if not args.augmentations or "date_swap" in args.augmentations:
        if args.load_intermediate:
            data_dateswp = load_data(args, "dateswp")
            # for example in data_dateswp:
            #     example['text'].text = ''
            print("Loaded %s date swap example pairs." % len(data_dateswp))

        if not data_dateswp:
            print("Creating date swap examples")
            dateswap_op = ops.DateSwap()
            data_dateswp = apply_transformation(data_positive, dateswap_op)
            # for example in data_dateswp:
            #     example['text'].text = ''
            print("DateSwap %s example pairs." % len(data_dateswp))

            if args.save_intermediate:
                save_data(args, data_dateswp, "dateswp")
    # data_dateswp = []

    data_numswp = []
    if not args.augmentations or "number_swap" in args.augmentations:
        if args.load_intermediate:
            data_numswp = load_data(args, "numswp")
            # for example in data_numswp:
            #     example['text'].text = ''
            print("Loaded %s number swap example pairs." % len(data_dateswp))

        if not data_numswp:
            print("Creating number swap examples")
            numswap_op = ops.NumberSwap()
            data_numswp = apply_transformation(data_positive, numswap_op)
            # for example in data_numswp:
            #     example['text'].text = ''
            print("NumberSwap %s example pairs." % len(data_numswp))

            if args.save_intermediate:
                save_data(args, data_numswp, "numswp")
    # data_numswp = []

    data_entswp = []
    if not args.augmentations or "entity_swap" in args.augmentations:
        if args.load_intermediate:
            data_entswp = load_data(args, "entswp")
            # for example in data_entswp:
            #     example['text'].text = ''
            print("Loaded %s entity swap example pairs." % len(data_entswp))

        if not data_entswp:
            print("Creating entity swap examples")
            entswap_op = ops.EntitySwap()
            data_entswp = apply_transformation(data_positive, entswap_op)
            # for example in data_entswp:
            #     example['text'].text = ''
            print("EntitySwap %s example pairs." % len(data_entswp))

            if args.save_intermediate:
                save_data(args, data_entswp, "entswp")
    # data_entswp = []

    data_negation = []
    if not args.augmentations or "negation" in args.augmentations:
        if args.load_intermediate:
            data_negation = load_data(args, "negation")
            # for example in data_negation:
            #     example['text'].text = ''
            print("Loaded %s negation example pairs." % len(data_negation))

        if not data_negation:
            print("Creating negation examples")
            negation_op = ops.NegateSentences()
            data_negation = apply_transformation(data_positive, negation_op)
            # for example in data_negation:
            #     example['text'].text = ''
            print("Negation %s example pairs." % len(data_negation))

            if args.save_intermediate:
                save_data(args, data_negation, "negation")
    # data_negation = []

    data_negative = data_pronoun + data_dateswp + \
        data_numswp + data_entswp + data_negation
    save_data(args, data_negative, "negative")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("data_file", type=str,
                        help="Path to file containing source documents.")
    PARSER.add_argument("--augmentations", type=str, nargs="+",
                        default=(), help="List of data augmentation applied to data.")
    PARSER.add_argument("--translation_api", type=str, default='easynmt',
                        help="Translation API to be used (easynmt / google).")
    PARSER.add_argument("--all_augmentations", action="store_true",
                        help="Flag whether all augmentation should be applied.")
    PARSER.add_argument("--save_intermediate", action="store_true",
                        help="Flag whether intermediate data from each transformation should be saved in separate files.")
    PARSER.add_argument("--load_intermediate", action="store_true",
                        help="Flag whether intermediate data from each transformation should be loaded if available.")
    ARGS = PARSER.parse_args()
    main(ARGS)
