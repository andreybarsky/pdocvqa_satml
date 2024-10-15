import os
from utils import save_yaml


def save_model(model, epoch, kwargs, update_best=False):
    save_dir = os.path.join(kwargs.save_dir, 'checkpoints', "{:s}".format(kwargs.experiment_name))


    model.model.save_pretrained(os.path.join(save_dir, f"model__{epoch:.1f}.ckpt"))
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.processor if hasattr(model, 'processor') else None
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(save_dir, f"model__{epoch:.1f}.ckpt"))
    ckpt_name = f"model__{epoch:.1f}.ckpt"
    save_yaml(os.path.join(save_dir, ckpt_name, "experiment_config.yml"), kwargs)
    print(f'Saving model to: {os.path.join(save_dir, ckpt_name)}')

    if update_best:
        model.model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)
        print(f'Saving model to: {os.path.join(save_dir, "best.ckpt")}')
