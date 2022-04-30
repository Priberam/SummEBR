import argparse
import json
from tqdm import tqdm
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='predictions.jsonl')
    parser.add_argument('--scores_rouge', type=str, default=None)
    parser.add_argument('--scores_ctc', type=str, default=None)
    parser.add_argument('--scores_questeval', type=str, default=None)
    args = parser.parse_args()


    with open(args.predictions, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    predictions = [json.loads(line) for line in tqdm(lines)]

    top1acc, top3acc, badchoices = 0, 0, 0
    for pred in predictions:
        if pred['candidate_rank'] == 0:
            top1acc += 1.
        if pred['candidate_rank'] < 3:
            top3acc += 1.
        if pred['candidate_rank'] >= 4:
            badchoices += 1.

    top1acc /= len(predictions)
    top3acc /= len(predictions)
    badchoices /= len(predictions)

    predictions = [pred['summary_index'] for pred in predictions]

    rouge1_avg, rouge2_avg, rougel_avg = -1, -1 ,-1
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    if args.scores_rouge is not None:
        with open(args.scores_rouge, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        scores = [json.loads(line) for line in tqdm(lines)]
        scores.pop(-1)
        assert len(scores) == len(predictions), print(len(scores), len(predictions))

        rouge1_avg, rouge2_avg, rougel_avg = 0., 0., 0.
        for pred, score in zip(predictions, scores):
            rouge1_avg += score['rouge1'][pred]
            rouge2_avg += score['rouge2'][pred]
            rougel_avg += score['rougel'][pred]
            rouge1_scores.append(score['rouge1'][pred])
            rouge2_scores.append(score['rouge2'][pred])
            rougel_scores.append(score['rougel'][pred])
        rouge1_avg /= len(predictions)
        rouge2_avg /= len(predictions)
        rougel_avg /= len(predictions)

        with open('rouge1_scores.pkl', 'wb') as fd:
            pickle.dump(rouge1_scores, fd)
        with open('rouge2_scores.pkl', 'wb') as fd:
            pickle.dump(rouge2_scores, fd)
        with open('rougel_scores.pkl', 'wb') as fd:
            pickle.dump(rougel_scores, fd)

    ctc_sum_avg, ctc_relevance_avg, ctc_consistency_avg = -1, -1, -1
    ctc_relevance_scores, ctc_consistency_scores = [], []
    if args.scores_ctc is not None:
        with open(args.scores_ctc, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        scores = [json.loads(line) for line in tqdm(lines)]
        scores.pop(-1)
        assert len(scores) == len(predictions), print(len(scores), len(predictions))

        ctc_sum_avg, ctc_relevance_avg, ctc_consistency_avg = 0., 0., 0.
        for pred, score in zip(predictions, scores):
            ctc_relevance_avg += score['relevance'][pred]
            ctc_consistency_avg += score['consistency'][pred]
            ctc_sum_avg += score['relevance'][pred] + score['consistency'][pred]
            ctc_relevance_scores.append(score['relevance'][pred])
            ctc_consistency_scores.append(score['consistency'][pred])
        ctc_sum_avg /= len(predictions)
        ctc_relevance_avg /= len(predictions)
        ctc_consistency_avg /= len(predictions)

        with open('ctc_relevance_scores.pkl', 'wb') as fd:
            pickle.dump(ctc_relevance_scores, fd)
        with open('ctc_consistency_scores.pkl', 'wb') as fd:
            pickle.dump(ctc_consistency_scores, fd)

    questeval_avg = -1
    if args.scores_questeval is not None:
        with open(args.scores_questeval, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
        scores = [json.loads(line) for line in tqdm(lines)]
        scores.pop(-1)
        assert len(scores) == len(predictions), print(len(scores), len(predictions))

        questeval_avg = 0.
        questeval_scores = []
        for pred, score in zip(predictions, scores):
            questeval_avg += score['questeval'][pred]
            questeval_scores.append(score['questeval'][pred])
        questeval_avg /= len(predictions)

        with open('questeval_scores.pkl', 'wb') as fd:
            pickle.dump(questeval_scores, fd)

    print('R1 & R2 & RL & QE & Cons & Rel')
    print('${:.2f}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}$'.format(100*rouge1_avg, 100*rouge2_avg, 100*rougel_avg, 100*questeval_avg, 100*ctc_consistency_avg, 100*ctc_relevance_avg))
    print('top1acc:', top1acc)
    print('top3acc:', top3acc)
    print('badchoices:', badchoices)

if __name__ == '__main__':
    main()