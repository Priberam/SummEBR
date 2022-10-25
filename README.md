# EBR

Source code for the paper "Improving abstractive summarization via energy-based re-ranking" (D. Pernes, A. Mendes, and A. F. T. Martins).
Presented at the *2nd Workshop on Natural Language Generation, Evaluation, and Metrics* **(GEM 2022)**.

If you wish to use the code, please read the attached **LICENSE.md**.

## Training

Training the EBR model involves the following steps:

1. Sampling candidates from an abstractive summarization model (BART or PEGASUS):

2. Scoring and ranking candidates according to the desired metric.

3. Fine-tuning BERT model using the sampled candidates and the ranking loss.

We provide a customizable `train.sh` script that can be used to train the EBR model.

## Testing

We provide the model checkpoints and the candidate summaries used in the experimental evaluation.
To reproduce the experiments:

1. If you haven't trained the model and want to use the data and checkpoints we provide:

    1. Save data and checkpoints at `./data` and `./checkpoints`, respectively.

    2. Compute ROUGE, QuestEval, and CTC scores for the test data you want to evaluate. E.g.:

            python scorer.py --source=./data/cnndm/bart/diverse-samples-test.jsonl --results_rouge=./data/cnndm/bart/results-rougel-test.jsonl --results_questeval=./data/cnndm/bart/results-questeval-test.jsonl --results_ctc=./data/cnndm/bart/results-ctc-test.jsonl

2. Use the desired EBR model to rank the test candidates. E.g.:

        python run-ranker.py --do_predict --gpus=1 -d ./data/cnndm/bart --metric=ctc_sum --checkpoint=./checkpoints/cnndm/bart/ebr-ctc_sum.ckpt --predictions_file=./data/cnndm/ebr-ctc_sum-predictions.jsonl

3. Get the results. E.g.:

        python score-ranked.py --predictions=./data/cnndm/ebr-ctc_sum-predictions.jsonl --scores_rouge=./data/results-rougel-test.jsonl --scores_questeval=./data/results-questeval-test.jsonl --scores_ctc=./data/results-ctc-test.jsonl
