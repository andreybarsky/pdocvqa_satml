import json, random, copy, sys
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import numpy as np

from eval import evaluate
from logger import Logger
from metrics import Evaluator
from checkpoint import save_model
from datasets.PFL_DocVQA import collate_fn
from utils import seed_everything, load_config, parse_args, set_parameters_model, serialise_h5
from build_utils import build_model, build_dataset, build_optimizer, build_provider_dataset, build_centralized_optimizer
from differential_privacy.dp_utils import add_dp_noise, clip_parameters, flatten_params, get_shape, reconstruct_shape

def train(data_loader, model, optimizer, lr_scheduler, evaluator,
          logger, config, max_batches=None, use_logger=True):
    model.model.train()
    logger.current_epoch += 1
    agg_update = None

    total_training_steps = len(data_loader)
    total_training_samples = len(data_loader.dataset)
    pbar = tqdm(total=total_training_steps)

    for batch_idx, batch in enumerate(data_loader):

        gt_answers = batch['answers']
        outputs, pred_answers, _ = model.forward(batch, return_pred_answer=True)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        metric = evaluator.get_metrics(gt_answers, pred_answers)

        log_dict = {
            'Train/Batch loss': outputs.loss.item(),
            'Train/Batch Accuracy': np.mean(metric['accuracy']),
            'Train/Batch ANLS': np.mean(metric['anls']),
            'lr': optimizer.param_groups[0]['lr']
        }

        if use_logger:
            logger.logger.log(log_dict)
        pbar.update()

    pbar.close()

def train_dp(data_loaders, model, optimizer, evaluator,
             logger, config, use_logger=True):
    model.model.train()
    param_keys = list(model.model.state_dict().keys())
    parameters = copy.deepcopy(list(model.model.state_dict().values()))

    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else config.lora for n, p in model.model.state_dict().items()]

    agg_update = None
    if not config.use_dp and len(data_loaders) > 1:
        raise ValueError("Non private training should only use one data loader.")

    total_training_steps = sum([len(data_loader) for data_loader in data_loaders])
    total_training_samples = sum([len(data_loader.dataset) for data_loader in data_loaders])
    pbar = tqdm(total=total_training_steps, desc=f"Training Epoch {logger.current_epoch}")

    total_loss = 0
    epoch_acc = 0
    epoch_anls = 0

    steps_taken = 0

    for p, provider_dataloader in enumerate(data_loaders):
        # For every provider, set model weights to the beginning state of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        model.model.load_state_dict(state_dict, strict=True)
        model.model.train()

        # Reset the optimizer for every providers at every epochs
        if config.use_dp:
            optimizer = build_optimizer(model, config)

        prov_name = provider_dataloader.dataset.provider.replace('\n', '\\n')
        prov_short_name = prov_name[:20]

        # Perform N provider iterations (each provider has their own dataloader in the non-private case)
        for batch_idx, batch in enumerate(provider_dataloader):

            gt_answers = batch['answers']
            outputs, pred_answers, _ = model.forward(batch, return_pred_answer=True)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            metric = evaluator.get_metrics(gt_answers, pred_answers)

            total_loss += outputs.loss.item()
            epoch_acc += np.sum(metric['accuracy'])
            epoch_anls += np.sum(metric['anls'])

            log_dict = {
                'Train/Batch loss': outputs.loss.item(),
                'Train/Batch Accuracy': np.mean(metric['accuracy']),
                'Train/Batch ANLS': np.mean(metric['anls']),
                'lr': optimizer.param_groups[0]['lr']
            }

            steps_taken += 1

            if use_logger:
                logger.logger.log(log_dict)
            logger.current_epoch += 1


            pbar.set_description(f"{prov_short_name:>20} (P{p:>3}/{len(data_loaders):>3}): B{(batch_idx+1):>2}/{len(provider_dataloader):>2}")
            pbar.update()

        # After all the iterations for each provider:
        # Get the update
        new_update = [w - w_0 for w, w_0 in zip(list(model.model.state_dict().values()), parameters)]

        if config.use_dp:
            # flatten update
            shapes = get_shape(new_update)
            new_update = flatten_params(new_update)

            # clip update:
            old_norm = torch.linalg.vector_norm(new_update, ord=2).item()
            new_update = clip_parameters(new_update, clip_norm=config.dp_params.sensitivity)

            # Aggregate (Avg)
            if agg_update is None:
                agg_update = new_update
            else:
                agg_update += new_update

    # Handle DP after all updates are done
    if config.use_dp:
        # Add the noise to the sum of all provider's clipped update
        agg_update = add_dp_noise(agg_update, noise_multiplier=config.dp_params.noise_multiplier, sensitivity=config.dp_params.sensitivity)

        # Divide the noisy aggregated update by the number of providers (Average update).
        agg_update = torch.div(agg_update, len(data_loaders))

        # Add the noisy update to the original model
        agg_update = reconstruct_shape(agg_update, shapes)

        # Restore original weights (without noise) from frozen layers.
        agg_update = [upd if not is_frozen else 0 for upd, params, is_frozen in zip(agg_update, parameters, frozen_parameters)]

    else:
        agg_update = new_update

    upd_weights = [
        torch.add(agg_upd, w_0).cpu()
        for agg_upd, w_0, is_frozen in zip(agg_update, copy.deepcopy(parameters), frozen_parameters)
        if not is_frozen
    ]  # Send weights of NON-Frozen layers.

    pbar.close()

    epoch_log_dict = {
        'Train/Epoch loss': total_loss / total_training_samples,
        'Train/Epoch Accuracy': epoch_acc / total_training_samples,
        'Train/Epoch ANLS': epoch_anls / total_training_samples,
        'Epoch': logger.current_epoch
    }
    if use_logger:
        logger.logger.log(epoch_log_dict)

    updated_model = set_parameters_model(model, upd_weights, frozen_parameters)
    return updated_model

def main():
    CENTRALIZED = True

    args = parse_args()
    config = load_config(args)

    seed_everything(config.seed)

    model = build_model(config)

    evaluator = Evaluator(case_sensitive=False)

    use_logger = not config.no_logger
    if use_logger:
        logger = Logger(config)
        logger.log_model_parameters(model)
    else:
        logger = lambda:0 # generic struct that can accept attribute allocation
        logger.current_epoch = 0

    epochs = config.train_epochs

    if config.use_h5:
        # check if h5 image directory exists, and if not, create it:
        if not os.path.exists(config.images_h5_path):
            print(f'Requested h5 image path but path does not exist, so performing first-time setup and serialising to: {config.images_h5_path}')
            serialise_h5(config)

    if config.use_dp:
        # Pick a subset of providers
        provider2docidx = json.load(open(config.provider_docs, 'r'))
        # provider2client = json.load(open(config.provider_client, 'r'))

        provider_names = list(provider2docidx.keys()) # plain list of strings

        val_dataset = build_dataset(config, 'valid', use_h5_images=args.use_h5)
        val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = build_optimizer(model, config=config)

        config.return_scores_by_sample = False
        config.return_pred_answers = False

        if getattr(config, 'eval_start', False):
            logger.current_epoch = -1 # that is, before epoch 0
            accuracy, anls, _, _ = evaluate(val_data_loader, model, evaluator, config, epoch=logger.current_epoch)
            is_updated = evaluator.update_global_metrics(accuracy, anls, -1)
            if use_logger:
                logger.log_val_metrics(accuracy, anls, update_best=is_updated)
            logger.current_epoch += 1

        # Build dataloaders once
        train_datasets = []
        client_id = 0 # all clients are 0 in the centralized setting
        for p, provider in enumerate(provider2docidx.keys()):
            train_datasets.append(build_provider_dataset(config, 'train', provider2docidx, provider, client_id=0, use_h5_images=args.use_h5))


        all_train_data_loaders = np.array([DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn) for train_dataset in train_datasets])

        # approx number of training steps in 1 non-DP epoch, for eval:
        epoch_size = sum([int(np.ceil(len(train_dataset) // config.batch_size)) for train_dataset in train_datasets])


        steps_trained_since_eval = 0

        # note that under DP, these are not strictly 'epochs' but update iterations analogous to FL rounds
        for epoch_ix in range(epochs):
            # subsample providers at every iteration

            # poisson sampling:
            sampling_probability = config.dp_params.providers_per_iteration / len(all_train_data_loaders)
            sample_mask = np.random.choice(a=[False, True], size=len(all_train_data_loaders), p=[1-sampling_probability, sampling_probability])
            # in cases of very low sample probability, ensure at least 1 is sampled:
            if sum(sample_mask) == 0:
                # flip a random one to true:
                sample_mask[np.random.choice(np.arange(len(sample_mask)))] = True
            train_data_loaders = all_train_data_loaders[sample_mask]

            sample_idxs = np.where(sample_mask)[0]
            sampled_providers = [provider_names[i] for i in sample_idxs] # list of provider names

            batches_per_provider = np.asarray([len(train_loader) for train_loader in train_data_loaders])
            avg_batches = np.mean(batches_per_provider)


            if use_logger:
                logger.current_epoch = epoch_ix
                logger.sampled_providers[epoch_ix] = sampled_providers
                logger.log_subsample() # save sampled providers to json in logger directory
                logger.current_epoch = epoch_ix


            model = train_dp(train_data_loaders, model, optimizer, evaluator, logger, config, use_logger=use_logger)

            # training with DP only evaluates for every full 'epoch'-equivalent
            # (that is, after as many steps as in non-DP), not at every DP iteration
            this_epoch_training_steps = sum([len(data_loader) for data_loader in train_data_loaders])
            steps_trained_since_eval += this_epoch_training_steps
            if (steps_trained_since_eval >= epoch_size) or (epoch_ix == epochs-1): # ensure evaluation at the very end
                accuracy, anls, _, _ = evaluate(val_data_loader, model, evaluator, config, epoch_ix)
                is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
                if use_logger:
                    logger.log_val_metrics(accuracy, anls, update_best=is_updated)
                save_model(model, epoch_ix, config, update_best=is_updated)
                # tick over the counter:
                steps_trained_since_eval -= epoch_size


    else:
        train_dataset = build_dataset(config, 'train', use_h5_images=args.use_h5)
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

        val_dataset = build_dataset(config, 'valid', use_h5_images=args.use_h5)
        print(f'Validating on: {val_dataset.imdb_path}')
        val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        logger.len_dataset = len(train_data_loader)
        optimizer, lr_scheduler = build_centralized_optimizer(model, length_train_loader=len(train_data_loader), config=config)

        config.return_scores_by_sample = False
        config.return_pred_answers = False

        if getattr(config, 'eval_start', False):
            logger.current_epoch = -1
            accuracy, anls, _, _ = evaluate(val_data_loader, model, evaluator, config, epoch=logger.current_epoch)
            is_updated = evaluator.update_global_metrics(accuracy, anls, -1)
            if use_logger:
                logger.log_val_metrics(accuracy, anls, update_best=is_updated)
            logger.current_epoch += 1

        for epoch_ix in range(epochs):
            logger.current_epoch = epoch_ix

            train(train_data_loader, model, optimizer, lr_scheduler, evaluator, logger, use_logger=use_logger, config=config)

            accuracy, anls, _, _ = evaluate(val_data_loader, model, evaluator, config, epoch_ix)

            is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
            if use_logger:
                logger.log_val_metrics(accuracy, anls, update_best=is_updated)
            save_model(model, epoch_ix, config, update_best=is_updated)

if __name__ == "__main__":
    outp = main()
