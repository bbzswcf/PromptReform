# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse
from bleu import _bleu
import json
from codebleu import calc_codebleu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', default='prompt_reformulation/dataset/test.jsonl', help="filename of the labels, in jsonl format.")
    parser.add_argument('--predictions', default='code_generation/results/test_pred2.txt', help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()
    gt_path = 'code_generation/evaluator/ground_truth.txt'
    preds = open(args.predictions, "r", encoding='utf-8').readlines()
    gts = open(args.answers, "r", encoding='utf-8').readlines()

    # assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    prediction = []
    reference = []

    for pred in preds:
        pred = pred.strip()
        prediction.append(pred)
        
    with open(gt_path, "w", encoding='utf-8') as wf:
        idx=0
        for gt in gts:
            idx+=1
            gt = json.loads(gt)["function"]
            reference.append(gt)
            wf.write(gt.replace("\n", r"\n").replace("\r", r"\r")+"\n")

    bleu_score = round(_bleu(gt_path, args.predictions), 2)
    codebleu_score = round(calc_codebleu(reference, prediction, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None), 2)
    logger.info(f"BLEU: {bleu_score}, CodeBLEU: {codebleu_score}")


if __name__ == "__main__":
    main()
