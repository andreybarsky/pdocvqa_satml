import torch
import numpy as np
import transformers
from transformers import get_scheduler

def build_optimizer(model, config):
    optimizer_class = getattr(torch.optim, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config.lr))

    return optimizer

def build_centralized_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config.lr))
    num_training_steps = config.train_epochs * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config.warmup_iterations, num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler

def build_model(config):

    available_models = ['t5', 'vt5']
    if config.model_name.lower() == 't5':
        from models.T5 import T5
        model = T5(config)

    elif config.model_name.lower() == 'vt5':
        from models.VT5 import VT5
        model = VT5(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose one of {:}".format(config.model_name, ', '.join(available_models)))

    model.model.to(config.device)
    return model

def build_dataset(config, split, client_id=None, use_h5_images=False, **kwargs):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['use_images'] = True

    if client_id is not None:
        dataset_kwargs['client_id'] = client_id

    # Build dataset
    if 'DocVQA' in config.dataset_name:
    # if config.dataset_name == 'PFL-DocVQA':
        from datasets.PFL_DocVQA import PFL_DocVQA
        h5_img_path = config.images_h5_path if use_h5_images else None
        img_dir = config.images_dir if hasattr(config, 'images_dir') else None

        dataset = PFL_DocVQA(config.imdb_dir, img_dir, split, dataset_kwargs,
                             h5_img_path=h5_img_path, **kwargs)

    else:
        raise ValueError

    return dataset

def build_dataset_of_subset_of_providers(config, split, provider2doc, list_of_providers, client_id=None, use_h5_images=False, **kwargs):
    """
    This function builds a dataset for a subset of providers.
    """

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['use_images'] = True

    if client_id:
        dataset_kwargs['client_id'] = client_id

    h5_img_path = config.images_h5_path if use_h5_images else None
    img_dir = config.images_dir if hasattr(config, 'images_dir') else None

    # Build dataset by looping through all providers
    # and collecting the indexes of the documents for each provider
    list_of_indicies = []
    for provider in list_of_providers:
        provider_indexes = provider2doc[provider]
        list_of_indicies.extend(provider_indexes)
    

    assert 0 not in list_of_indicies
    if 'DocVQA' in config.dataset_name:
        from datasets.PFL_DocVQA import PFL_DocVQA
        dataset = PFL_DocVQA(config.imdb_dir, img_dir, split, dataset_kwargs,
            list_of_indicies, h5_img_path=h5_img_path, **kwargs)

    else:
        raise ValueError

    return dataset

def build_provider_dataset(config, split, provider2doc, provider, client_id=None, use_h5_images=False, **kwargs):
    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config.model_name.lower() in ['vt5']:
        dataset_kwargs['use_images'] = True

    if client_id:
        dataset_kwargs['client_id'] = client_id

    h5_img_path = config.images_h5_path if use_h5_images else None
    img_dir = config.images_dir if hasattr(config, 'images_dir') else None

    # Build dataset
    indexes = provider2doc[provider]
    assert 0 not in indexes
    if 'DocVQA' in config.dataset_name:
        from datasets.PFL_DocVQA import PFL_DocVQA
        dataset = PFL_DocVQA(config.imdb_dir, img_dir, split, dataset_kwargs,
            indexes, h5_img_path=h5_img_path, **kwargs)

    else:
        raise ValueError

    dataset.provider = provider
    return dataset
