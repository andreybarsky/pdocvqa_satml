import os, ast, yaml, json, random, datetime, argparse
import torch
import numpy as np
import editdistance

def parse_args():
    parser = argparse.ArgumentParser(description="PFL-DocVQA Centralized Trainng")

    # Required:
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to yml file with model configuration.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to yml file with dataset configuration.')

    # DP training:
    parser.add_argument('-dp', '--use_dp', action='store_true', default=False, help='Use Differential Privacy.')


    # Optional
    parser.add_argument('--eval-start', action='store_true', default=False, help='Whether to evaluate the model before training or not.')
    parser.add_argument('-nl', '--no_logger', action='store_true', default=False, help='Disable WandB logger')


    # For faster training:
    parser.add_argument('-h5', '--use_h5', action='store_true', default=False, help='Use h5 image archive instead of decompressing jpegs')

    # Overwrite config parameters
    parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    parser.add_argument('-msl', '--max-sequence-length', type=int, help='Max input sequence length of the model.')
    parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')
    parser.add_argument('--save-dir', type=str, help='Checkpoints directory.')

    # Experimental:
    parser.add_argument('--lora', action='store_true', default=False, help='Run LoRA.')




    return parser.parse_args()

def parse_multitype2list_arg(argument):
    if argument is None:
        return argument

    if '-' in argument and '[' in argument and ']' in argument:
        first, last = argument.strip('[]').split('-')
        argument = list(range(int(first), int(last)))
        return argument

    argument = ast.literal_eval(argument)

    if isinstance(argument, int):
        argument = [argument]

    elif isinstance(argument, list):
        argument = argument

    return argument


def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)


"""
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
"""

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def check_config(config):

    if config.flower:
        assert config.fl_params.sample_clients <= config.fl_params.total_clients, "Number of sampled clients ({:d}) can't be greater than total number of clients ({:d})".format(config.fl_params.sample_clients, config.fl_params.total_clients)

    if 'save_dir' in config:
        if not config.save_dir.endswith('/'):
            config.save_dir = config.save_dir + '/'

        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
            os.makedirs(os.path.join(config.save_dir, 'results'))
            os.makedirs(os.path.join(config.save_dir, 'communication_logs'))

    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    config.experiment_name = "{:s}_{:s}{:s}__{:}".format(config.model_name, config.dataset_name, '_dp' if config.use_dp else '', experiment_date)

    return True

def load_config(args):
    model_config_path = "configs/models/{:}.yml".format(args.model)
    dataset_config_path = "configs/datasets/{:}.yml".format(args.dataset)
    model_config = parse_config(yaml.safe_load(open(model_config_path, "r")), args)
    dataset_config = parse_config(yaml.safe_load(open(dataset_config_path, "r")), args)
    training_config = model_config.pop('training_parameters')

    dp_config = model_config.pop('dp_parameters') if 'dp_parameters' in model_config and args.use_dp else None
    dp_keys = []
    if dp_config is not None:
        dp_config.update({k: v for k, v in args._get_kwargs() if k in dp_config and v is not None})
        dp_keys.extend(dp_config.keys())

    lora_config = model_config.pop('lora_parameters') if 'lora_parameters' in model_config and args.lora else None
    lora_keys = []
    if lora_config is not None:
        lora_config.update({k: v for k, v in args._get_kwargs() if k in lora_config and v is not None})
        lora_keys.extend(lora_config.keys())

    # Merge config values and input arguments.
    config = {**dataset_config, **model_config, **training_config}
    config = config | {k: v for k, v in args._get_kwargs() if v is not None}

    # Remove duplicate keys
    config.pop('model')
    config.pop('dataset')
    [config.pop(k) for k in list(config.keys()) if (k in dp_keys)]
    [config.pop(k) for k in list(config.keys()) if (k in lora_keys)]

    config = argparse.Namespace(**config)

    if dp_config is not None:
        config.dp_params = argparse.Namespace(**dp_config)

        # config['group_sampling_probability'] = config['client_sampling_probability'] * 50 / 340  # (Number of selected clients / total number of clients) * (Number of selected groups / MIN(number of groups among the clients))
        # config.dp_params.group_sampling_probability = config.dp_params.client_sampling_probability * config.dp_params.providers_per_fl_round / 340  # 0.1960  # config['client_sampling_probability'] * 50 / 340  # (Number of selected clients / total number of clients) * (Number of selected groups / MIN(number of groups among the clients))

    if lora_config is not None:
        config.lora_params = argparse.Namespace(**lora_config)

    # Set default seed
    if 'seed' not in config:
        print("Seed not specified. Setting default seed to '{:d}'".format(42))
        config.seed = 1026

    if 'save_dir' in config:
        if not config.save_dir.endswith('/'):
            config.save_dir = config.save_dir + '/'

        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
            os.makedirs(os.path.join(config.save_dir, 'results'))
            os.makedirs(os.path.join(config.save_dir, 'communication_logs'))

    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    config.experiment_name = "{:s}_{:s}{:s}{:s}__{:}".format(
        config.model_name, config.dataset_name, f'_dp_C{config.dp_params.sensitivity:.1f}_e{config.dp_params.noise_multiplier:.3f}' if config.use_dp else '',
        '_lora' if config.lora else '', experiment_date
    )

    return config

def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get('includes', []):
        config = load_config(included_config_path, args) | config

    return config

def correct_alignment(context, answer, start_idx, end_idx):
    if context[start_idx: end_idx] == answer:
        return [start_idx, end_idx]

    elif context[start_idx - 1: end_idx] == answer:
        return [start_idx - 1, end_idx]

    elif context[start_idx: end_idx + 1] == answer:
        return [start_idx, end_idx + 1]

    else:
        print(context[start_idx: end_idx], answer)
        return None

def time_stamp_to_hhmmss(timestamp, string=True):
    hh = int(timestamp/3600)
    mm = int((timestamp-hh*3600)/60)
    ss = int(timestamp - hh*3600 - mm*60)

    time = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss) if string else [hh, mm, ss]

    return time

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def merge_data(data_dir):
    brokens = { "client_9": ["042e71e9638024a37aa9eb8f_3"]}
    doc2provider = json.load(open(os.path.join(data_dir, 'doc2provider.json'), 'r'))
    data_points = json.load(open(os.path.join(data_dir, 'data_points.json'), 'r'))

    print(f"Read client 0 data")
    imdb_train = np.load(os.path.join(data_dir, 'imdb_train_client_0.npy'), allow_pickle=True)
    for client_id in range(1, 10):
        print(f"Read client {client_id} data")
        data_npy = np.load(os.path.join(data_dir, f'imdb_train_client_{client_id}.npy'), allow_pickle=True)
        if f"client_{client_id}" in brokens:
            broken_inds = []
            for broken_im in brokens[f"client_{client_id}"]:
                docid = broken_im.split("_")[0]
                provider = doc2provider[docid]
                broken_inds.extend([_ind for _ind in data_points[f"client_{client_id}"][provider] if data_npy[_ind]["image_name"] == broken_im])
            non_broken_inds = np.array([True] * len(data_npy))
            non_broken_inds[broken_inds] = False
            data_npy = data_npy[non_broken_inds]
            if len(broken_inds) > 0:
                np.save(open(os.path.join(data_dir, f'imdb_train_client_{client_id}.npy'), 'wb'), data_npy, allow_pickle=True)
        imdb_train = np.concatenate((imdb_train, data_npy[1:]))
    with open(os.path.join(data_dir, 'imdb_train.npy'), 'wb') as f:
        np.save(f, imdb_train, allow_pickle=True)

    centralized_data_points = dict()
    for _ind, _imdb in enumerate(imdb_train):
        if _ind == 0: continue;
        image_name = _imdb['image_name']
        docid = image_name.split('_')[0]
        provider = doc2provider[docid]
        if provider in centralized_data_points:
            centralized_data_points[provider].append(_ind)
        else:
            centralized_data_points[provider] = [_ind]
    with open(os.path.join(data_dir, 'centralized_data_points.json'), 'w') as f:
        json.dump(centralized_data_points, f)

def find_broken(images_dir, data_dir):
    from PIL import Image
    print(f"Read client 0 data")
    imdb_train = np.load(os.path.join(data_dir, 'imdb_train_client_0.npy'), allow_pickle=True)
    for client_id in range(1, 10):
        print(f"Read client {client_id} data")
        data_npy = np.load(os.path.join(data_dir, f'imdb_train_client_{client_id}.npy'), allow_pickle=True)
        imdb_train = np.concatenate((imdb_train, data_npy[1:]))

    imdb_train = imdb_train[1:]

    train_images = set()
    for _imdb in imdb_train:
        image_name = _imdb['image_name']
        train_images.add(image_name)
    train_images = list(train_images)
    print(f"Total train images: {len(train_images)}")

    broken = []
    for train_image in train_images:
        try:
            image = Image.open(os.path.join(images_dir, f"{train_image}.jpg")).convert("RGB")
        except Exception as e:
            print(f"Image file: {train_image} exception {e}")
            broken.append(train_image)
            continue
    print("Total broken: ", len(broken))

def set_parameters_model(model, parameters, frozen_parameters):
    i = 0
    params_dict = model.model.state_dict()
    for key, is_frozen in zip(model.model.state_dict().keys(), frozen_parameters):

        # Update state dict with new params.
        if not is_frozen:
            params_dict[key] = torch.Tensor(parameters[i])
            i += 1

    model.model.load_state_dict(params_dict, strict=True)
    return

def anls(ans, pred, thresh=0.5):
    ans = [_an.lower().strip() for _an in ans]
    pred = pred.lower().strip()

    if len(pred) == 0: return 0;

    maxsim = max([1 - editdistance.eval(_an, pred) / max(len(_an), len(pred)) for _an in ans])
    anls = maxsim if maxsim >= thresh else 0
    return anls

def accuracy(ans, pred):
    ans = [_an.lower().strip() for _an in ans]
    pred = pred.lower().strip()

    if len(pred) == 0: return 0;

    for _an in ans:
        if _an == pred:
            return 1
    return 0

def plot_roc_curve_multi(labels, ranks, legends, title, name):
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    save_path = os.path.join(f"roc_curve_{name.replace(' ', '_')}.png")
    for _i, (_y, _rank, _legend) in enumerate(zip(labels, ranks, legends)):
        # TPR/FPR
        fpr, tpr, _ = roc_curve(_y, _rank)
        tpr_fpr = tpr[np.where(fpr<.01)[0][-1]]

        # Balanced Accuracy
        bal_acc = 1 - (fpr+(1-tpr))/2

        # F1 Score
        n_pos = n_neg = len(_rank) // 2
        tp = tpr * n_pos; fn = n_pos - tp
        tn = (1 - fpr) * n_neg; fp = n_neg - tn
        pre = tp / (tp + fp); rec = tp / (tp + fn)
        f1 = (2 * pre * rec) / (pre + rec)

        print(f"MIA Results: {_legend}")
        print(f"{'TPR@FPR=0.01':>25} = {tpr_fpr*100:.2f}")
        print(f"{'(MAX) Balanced Accuracy':>25} = {np.max(bal_acc)*100:.2f}")
        print(f"{'(MAX) F1':>25} = {f1[np.argmax(bal_acc)]*100:.2f}")

        if _i == 0:
            plt.plot(fpr, fpr, '--', color='gray', label="Chance level, ACC=50%")
        plt.plot(fpr, tpr, label=f"{legends[_i]}, ACC={np.max(bal_acc)*100:.2f}%")

    plt.grid(); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.semilogx(); plt.semilogy(); plt.xlim(1e-3, 1); plt.ylim(1e-3, 1)
    plt.title(title); plt.legend(loc='lower right'); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")
