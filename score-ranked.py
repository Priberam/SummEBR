import argparse
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='predictions.jsonl')
    parser.add_argument('--scores_rougel', type=str, default=None)
    parser.add_argument('--scores_ctc', type=str, default=None)
    parser.add_argument('--scores_questeval', type=str, default=None)
    args = parser.parse_args()


    with open(args.predictions, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    predictions = [json.loads(line) for line in tqdm(lines)]

    top1acc, top3acc = 0, 0
    for pred in predictions:
        if pred['candidate_rank'] == 0:
            top1acc += 1.
        if pred['candidate_rank'] < 3:
            top3acc += 1.
    top1acc /= len(predictions)
    top3acc /= len(predictions)

    predictions = [pred['summary_index'] for pred in predictions]

    rougel_avg = None
    if args.scores_rougel is not None:
        with open(args.scores_rougel, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        scores = [json.loads(line) for line in tqdm(lines)]
        scores.pop(-1)
        assert len(scores) == len(predictions)

        rougel_avg = 0.
        for pred, score in zip(predictions, scores):
            rougel_avg += score['rougeL'][pred]
        rougel_avg /= len(predictions)

    ctc_sum_avg, ctc_relevance_avg, ctc_consistency_avg = None, None, None
    if args.scores_ctc is not None:
        with open(args.scores_ctc, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        scores = [json.loads(line) for line in tqdm(lines)]
        scores.pop(-1)
        assert len(scores) == len(predictions)

        ctc_sum_avg, ctc_relevance_avg, ctc_consistency_avg = 0., 0., 0.
        for pred, score in zip(predictions, scores):
            ctc_relevance_avg += score['relevance'][pred]
            ctc_consistency_avg += score['consistency'][pred]
            ctc_sum_avg += score['relevance'][pred] + score['consistency'][pred]
        ctc_sum_avg /= len(predictions)
        ctc_relevance_avg /= len(predictions)
        ctc_consistency_avg /= len(predictions)

    questeval_avg = None
    if args.scores_questeval is not None:
        with open(args.scores_questeval, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        scores = [json.loads(line) for line in tqdm(lines)]
        scores.pop(-1)
        assert len(scores) == len(predictions)

        questeval_avg = 0.
        for pred, score in zip(predictions, scores):
            questeval_avg += score['questeval'][pred]
        questeval_avg /= len(predictions)

    print('rougeL:', rougel_avg)
    print('ctc_sum:', ctc_sum_avg)
    print('ctc_relevance:', ctc_relevance_avg)
    print('ctc_consistency:', ctc_consistency_avg)
    print('questeval:', questeval_avg)
    print('top1acc:', top1acc)
    print('top3acc:', top3acc)

if __name__ == '__main__':
    main()