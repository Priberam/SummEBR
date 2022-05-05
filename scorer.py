from ctc_score import SummarizationScorer
from questeval.questeval_metric import QuestEval
from datasets import load_metric
import argparse
import numpy as np
import json
import nltk
from tqdm import tqdm
from utils import postprocess_text, replace_special_chars


ctc_scorer = SummarizationScorer(align='D-cnndm')
questeval_scorer = QuestEval(task='summarization', do_weighter=True)
rouge_scorer = load_metric('rouge')

def compute_rouge_score(reference, predictions):
    references = postprocess_text([reference])*len(predictions)
    predictions = postprocess_text(predictions)
    rougel_scores = rouge_scorer.compute(predictions=predictions,
                                         references=references,
                                         use_stemmer=True, use_agregator=False)['rougeLsum']
    rougel_scores = np.array([x.fmeasure for x in rougel_scores])

    rouge_scores = rouge_scorer.compute(predictions=predictions,
                                         references=references,
                                         use_stemmer=True, use_agregator=False)

    rouge1_scores = np.array([x.fmeasure for x in rouge_scores['rouge1']])
    rouge2_scores = np.array([x.fmeasure for x in rouge_scores['rouge2']])
    rougel_scores = np.array([x.fmeasure for x in rouge_scores['rougeLsum']])

    return rouge1_scores, rouge2_scores, rougel_scores

def compute_ctc_score(inpt, reference, predictions):
    inpt = replace_special_chars(inpt)
    reference = replace_special_chars(reference)
    predictions = [replace_special_chars(pred) for pred in predictions]

    consistency_scores, relevance_scores = [], []
    for pred in predictions:
        try:
            consistency = ctc_scorer.score(doc=inpt, refs=[], hypo=pred, aspect='consistency')
            relevance = ctc_scorer.score(doc=inpt, refs=[reference], hypo=pred, aspect='relevance')
            consistency_scores.append(consistency if consistency is not None else 0)
            relevance_scores.append(relevance if relevance is not None else 0)
        except:
            consistency_scores.append(0)
            relevance_scores.append(0)
            print('Couldn\'t compute CTC scores for the current example. Skipping it.')

    if not consistency_scores:
        consistency_scores = np.zeros(len(predictions))
    if not relevance_scores:
        relevance_scores = np.zeros(len(predictions))

    return np.array(consistency_scores), np.array(relevance_scores)

def compute_questeval_score(inpt, predictions):
    inpt = replace_special_chars(inpt)
    predictions = [replace_special_chars(pred) for pred in predictions]

    sources = [inpt]*len(predictions)
    qe_scores = questeval_scorer.corpus_questeval(
                hypothesis=predictions,
                sources=sources)['ex_level_scores']
    return np.array(qe_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='predictions.jsonl')
    parser.add_argument('--results_rouge', type=str, default='results-rouge.jsonl')
    parser.add_argument('--results_ctc', type=str, default='results-ctc.jsonl')
    parser.add_argument('--results_questeval', type=str, default='results-questeval.jsonl')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--skip_rouge', dest='skip_rouge', action='store_true')
    parser.add_argument('--skip_ctc', dest='skip_ctc', action='store_true')
    parser.add_argument('--skip_questeval', dest='skip_questeval', action='store_true')
    args = parser.parse_args()

    with open(args.source, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    examples = [json.loads(line) for line in tqdm(lines)]
    num_examples = (min(args.num_samples, len(examples)) if args.num_samples is not None and args.num_samples > 0
                    else len(examples))

    if not args.skip_rouge:
        print('Computing ROUGE scores...')
        with open(args.results_rouge, 'w', encoding='utf-8') as f:
            rouge1_med_avg, rouge1_amp_avg = 0, 0
            rouge2_med_avg, rouge2_amp_avg = 0, 0
            rougel_med_avg, rougel_amp_avg, rougel_best_avg = 0., 0., 0.
            for i, example in tqdm(enumerate(examples), total=num_examples+args.offset):
                if i < args.offset:
                    continue
                if i == num_examples + args.offset:
                    break

                predictions = [example[x] for x in example if 'gen_summary' in x]
                rouge1_scores, rouge2_scores, rougel_scores = compute_rouge_score(example['gold_summary'], predictions)
                rougel_rank = np.argsort(-rougel_scores)
                scores = {
                    'rouge1': rouge1_scores.tolist(),
                    'rouge2': rouge2_scores.tolist(),
                    'rougel': rougel_scores.tolist(),
                    'rank': rougel_rank.tolist(),
                }
                f.write(json.dumps(scores, ensure_ascii=False) + "\n")

                rouge1_med = np.median(rouge1_scores)
                rouge1_amp = np.amax(rouge1_scores) - np.amin(rouge1_scores)
                rouge2_med = np.median(rouge2_scores)
                rouge2_amp = np.amax(rouge2_scores) - np.amin(rouge2_scores)
                rougel_med = np.median(rougel_scores)
                rougel_amp = (rougel_scores[rougel_rank[0]] - rougel_scores[rougel_rank[-1]]) / 2.
                rougel_best = rougel_scores[rougel_rank[0]]

                rouge1_med_avg += rouge1_med
                rouge1_amp_avg += rouge1_amp
                rouge2_med_avg += rouge2_med
                rouge2_amp_avg += rouge2_amp
                rougel_med_avg += rougel_med
                rougel_amp_avg += rougel_amp
                rougel_best_avg += rougel_best

            rouge1_med_avg /= num_examples
            rouge1_amp_avg /= num_examples
            rouge2_med_avg /= num_examples
            rouge2_amp_avg /= num_examples

            rougel_med_avg /= num_examples
            rougel_amp_avg /= num_examples
            rougel_best_avg /= num_examples

            score_summary = {
                'rouge1_avg_median': rouge1_med_avg,
                'rouge1_avg_amplitude': rouge1_amp_avg,
                'rouge2_avg_median': rouge2_med_avg,
                'rouge2_avg_amplitude': rouge2_amp_avg,
                'rougel_avg_median': rougel_med_avg,
                'rougel_avg_amplitude': rougel_amp_avg,
                'rougel_avg_best': rougel_best_avg,
            }
            f.write(json.dumps(score_summary, ensure_ascii=False) + '\n')
            print(score_summary)

    if not args.skip_ctc:
        print('Computing CTC scores...')
        with open(args.results_ctc, 'w', encoding='utf-8') as f:
            consistency_med_avg, relevance_med_avg = 0., 0.
            consistency_amp_avg, relevance_amp_avg = 0., 0.
            consistency_best_avg, relevance_best_avg = 0., 0.
            consistency_sum_best_avg, relevance_sum_best_avg = 0., 0.
            for i, example in tqdm(enumerate(examples), total=num_examples+args.offset):
                if i < args.offset:
                    continue
                if i == num_examples + args.offset:
                    break

                predictions = [example[x] for x in example if 'gen_summary' in x]
                consistency_scores, relevance_scores = compute_ctc_score(example['text'], example['gold_summary'], predictions)
                consistency_rank = np.argsort(-consistency_scores)
                relevance_rank = np.argsort(-relevance_scores)
                sum_rank = np.argsort(-consistency_scores-relevance_scores)
                scores = {
                    'consistency': consistency_scores.tolist(),
                    'relevance': relevance_scores.tolist(),
                    'rank_consistency': consistency_rank.tolist(),
                    'rank_relevance': relevance_rank.tolist(),
                    'rank_sum': sum_rank.tolist(),
                }
                f.write(json.dumps(scores, ensure_ascii=False) + '\n')

                consistency_med = np.median(consistency_scores)
                consistency_amp = (consistency_scores[consistency_rank[0]] - consistency_scores[consistency_rank[-1]]) / 2.
                consistency_best = consistency_scores[consistency_rank[0]]
                consistency_sum_best = consistency_scores[sum_rank[0]]
                relevance_med = np.median(relevance_scores)
                relevance_amp = (relevance_scores[relevance_rank[0]] - relevance_scores[relevance_rank[-1]]) / 2.
                relevance_best = relevance_scores[relevance_rank[0]]
                relevance_sum_best = relevance_scores[sum_rank[0]]

                consistency_med_avg += consistency_med
                consistency_amp_avg += consistency_amp
                consistency_best_avg += consistency_best
                consistency_sum_best_avg += consistency_sum_best
                relevance_med_avg += relevance_med
                relevance_amp_avg += relevance_amp
                relevance_best_avg += relevance_best
                relevance_sum_best_avg += relevance_sum_best

            consistency_med_avg /= num_examples
            consistency_amp_avg /= num_examples
            consistency_best_avg /= num_examples
            consistency_sum_best_avg /= num_examples
            relevance_med_avg /= num_examples
            relevance_amp_avg /= num_examples
            relevance_best_avg /= num_examples
            relevance_sum_best_avg /= num_examples

            score_summary = {
                'consistency_avg_median': consistency_med_avg,
                'consistency_avg_amplitude': consistency_amp_avg,
                'consistency_avg_best': consistency_best_avg,
                'consistency_avg_sum_best': consistency_sum_best_avg,
                'relevance_avg_median': relevance_med_avg,
                'relevance_avg_amplitude': relevance_amp_avg,
                'relevance_avg_best': relevance_best_avg,
                'relevance_avg_sum_best': relevance_sum_best_avg,
            }
            f.write(json.dumps(score_summary, ensure_ascii=False) + "\n")
            print(score_summary)

    if not args.skip_questeval:
        print('Computing QuestEval scores...')
        with open(args.results_questeval, 'w', encoding='utf-8') as f:
            qe_med_avg, qe_amp_avg, qe_best_avg = 0., 0., 0.
            for i, example in tqdm(enumerate(examples), total=num_examples+args.offset):
                if i < args.offset:
                    continue
                if i == num_examples + args.offset:
                    break

                predictions = [example[x] for x in example if 'gen_summary' in x]
                qe_scores = compute_questeval_score(example['text'], predictions)
                qe_rank = np.argsort(-qe_scores)
                scores = {
                    'questeval': qe_scores.tolist(),
                    'rank': qe_rank.tolist(),
                }
                f.write(json.dumps(scores, ensure_ascii=False) + "\n")

                qe_med = np.median(qe_scores)
                qe_amp = (qe_scores[qe_rank[0]] - qe_scores[qe_rank[-1]]) / 2.
                qe_best = qe_scores[qe_rank[0]]

                qe_med_avg += qe_med
                qe_amp_avg += qe_amp
                qe_best_avg += qe_best

            qe_med_avg /= num_examples
            qe_amp_avg /= num_examples
            qe_best_avg /= num_examples

            score_summary = {
                'questeval_avg_median': qe_med_avg,
                'questeval_avg_amplitude': qe_amp_avg,
                'questeval_avg_best': qe_best_avg,
            }
            f.write(json.dumps(score_summary, ensure_ascii=False) + '\n')
            print(score_summary)

if __name__ == '__main__':
    main()