import torch
import secrets
import numpy as np
from tqdm import tqdm
import secrets

# secure prng:
csprng = secrets.SystemRandom()




def flatten_params(parameters):
    """
    Flat the list of tensors (layer params) into a single vector.
    """
    return torch.cat([torch.flatten(layer_norm) for layer_norm in parameters])


def clip_parameters(parameters, clip_norm):
    """
    Clip update parameters to clip norm.
    """
    current_norm = torch.linalg.vector_norm(parameters, ord=2)
    return torch.div(parameters, torch.max(torch.tensor(1, device=parameters.device), torch.div(current_norm, clip_norm)))


def get_shape(update: list):
    """
    Return a list of shapes given a list of tensors.
    """
    shapes = [ele.shape for ele in update]
    return shapes


def reconstruct_shape(flat_update, shapes):
    """
    Reconstruct the original shapes of the tensors list.
    """
    ind = 0
    rec_upd = []
    for shape in shapes:
        num_elements = torch.prod(torch.tensor(shape)).item()
        rec_upd.append(flat_update[ind:ind+num_elements].reshape(shape))
        ind += num_elements

    return rec_upd


def add_dp_noise(data, noise_multiplier, sensitivity):
    """
    Add differential privacy noise to data.
    """
    dp_noise = torch.normal(mean=0, std=noise_multiplier * sensitivity, size=data.shape, device=data.device)
    # add in place to save memory:
    return data.add_(dp_noise)



def add_dp_noise_csprng(data, noise_multiplier, sensitivity):
    """
    Add differential privacy noise to data using cryptographically-secure PRNG.
    Currently generates the noise in a loop on CPU, which is inefficient,
    but runs only once per provider per round, so not a major slowdown.
    """
    dp_noise = np.zeros(data.numel())
    for i in tqdm(range(data.numel())):
        dp_noise[i] = csprng.normalvariate(mu=0, sigma=noise_multiplier * sensitivity)

    # rand_noise = np.asarray([csprng.normalvariate(mu=0, sigma=noise_multiplier * sensitivity) for d in range(data.numel())])
    dp_noise_tensor = torch.tensor(dp_noise, device=data.device).reshape(data.shape)
    # add in place to save memory:
    data.add_(dp_noise_tensor)
    return data
