"""Score LongBench predictions with the metrics of paper Table 5.

Adapted from the LongBench evaluator (Bai et al., ACL 2024;
https://github.com/THUDM/LongBench, MIT). For each run folder in ``pred/`` the
script scores each ``<task>.jsonl`` with the metric of the task (F1, ROUGE-L,
accuracy or edit similarity; see ``metrics.py``). Each prediction gets the
maximum score over its reference answers. The script writes ``result.json``
(one score per task, 0-100) and ``list_result.json`` (scores per sample). It
skips a task that does not have all samples.

Usage::

    python eval.py                   # all runs in pred/
    python eval.py --run <folder>    # one run
"""
import argparse
import json
import os

from metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)

# Expected number of samples per task.
dataset_samples = {
    "narrativeqa": 200,
    "qasper":200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "passage_retrieval_en": 200,
    "passage_count": 200,
    "lcc": 500,
    "repobench-p": 500,
}

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pred_dir", default="pred", help="folder holding one sub-folder per run")
    parser.add_argument("--run", default=None, help="score only this run folder")
    return parser.parse_args(args)


def scorer(dataset, predictions, answers, all_classes):
    """Return the mean score of a task (0-100) and the list of per-sample scores."""
    score_list = []
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        score_list.append(score)
        total_score += score
    return round(100 * total_score / len(predictions), 2),score_list

def score_run(path):
    """Score each complete task of one run folder.

    Returns:
        A tuple ``({task: score}, {task: [per-sample scores]})``.
    """
    scores, per_sample = {}, {}
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".jsonl"):
            continue
        task = filename.split(".")[0]
        predictions, answers, all_classes = [], [], None
        with open(os.path.join(path, filename), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                predictions.append(row["pred"])
                answers.append(row["answers"])
                all_classes = row["all_classes"]
        expected = dataset_samples.get(task, 200)
        if len(predictions) != expected:
            print(f"skip {path}/{filename}: {len(predictions)} samples, expected {expected}")
            continue
        scores[task], per_sample[task] = scorer(task, predictions, answers, all_classes)
    return scores, per_sample


def main():
    """Score each run folder and write its result files."""
    args = parse_args()
    runs = [args.run] if args.run else sorted(os.listdir(args.pred_dir))
    for run in runs:
        path = os.path.join(args.pred_dir, run)
        if not os.path.isdir(path):
            continue
        scores, per_sample = score_run(path)
        print(f"{run}: {scores}")
        with open(os.path.join(path, "result.json"), "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        with open(os.path.join(path, "list_result.json"), "w") as f:
            json.dump(per_sample, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
