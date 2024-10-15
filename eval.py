import os, time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.PFL_DocVQA import collate_fn

from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset
from checkpoint import save_model

def evaluate(data_loader, model, evaluator, config, epoch, max_steps=None):
    return_scores_by_sample = getattr(config, 'return_scores_by_sample', False)
    return_answers = getattr(config, 'return_answers', False)

    if return_scores_by_sample:
        scores_by_samples = {}
        total_accuracies = []
        total_anls = []
    else:
        total_accuracies = 0
        total_anls = 0
    all_pred_answers = []

    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating Epoch {epoch}")):
        bs = len(batch['question_id'])
        with torch.no_grad():
            outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)

        metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))

        if return_scores_by_sample:
            for batch_idx in range(bs):
                scores_by_samples[batch['question_id'][batch_idx]] = {
                    'accuracy': metric['accuracy'][batch_idx],
                    'anls': metric['anls'][batch_idx],
                    'pred_answer': pred_answers[batch_idx],
                    'pred_answer_conf': answer_conf[batch_idx]
                }

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])

        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])

        if return_answers:
            all_pred_answers.extend(pred_answers)

        if max_steps is not None and batch_idx >= max_steps:
            # print(f'Manually aborting validation after {max_steps} batches')
            break


    if not return_scores_by_sample:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)
        scores_by_samples = []
    return total_accuracies, total_anls, all_pred_answers, scores_by_samples


def main_eval(config):
    start_time = time.time()

    config.return_scores_by_sample = True
    config.return_answers = True

    dataset = build_dataset(config, 'val')
    sampler = None
    pin_memory = False

    val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, sampler=sampler)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, config, epoch=0)
    accuracy, anls = np.mean(accuracy_list), np.mean(anls_list)

    inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
    logger.log_val_metrics(accuracy, anls, update_best=False)

    save_data = {
        "Model": config.model_name,
        "Model_weights": config.model_weights,
        "Dataset": config.dataset_name,
        "Page retrieval": getattr(config, 'page_retrieval', '-').capitalize(),
        "Inference time": inf_time,
        "Mean accuracy": accuracy,
        "Mean ANLS": anls,
        "Scores by samples": scores_by_samples,
    }

    results_file = os.path.join(config.save_dir, 'results', config.experiment_name)
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args)

    main_eval(config)
