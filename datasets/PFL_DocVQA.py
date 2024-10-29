import os, random
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import h5py

### for 1.1:
# prefix = 'imdb'
### for 1.0:
# prefix = 'blue'

# for faster loading when multiple datasets are needed:
IMDB_CACHE = {}
H5_CACHE = {}

class PFL_DocVQA(Dataset):

    def __init__(self, imdb_dir, images_dir, split, kwargs, indexes=None, h5_img_path=None):

        imdb_npy_path = os.path.join(imdb_dir, f"{split}.npy")

        # load imdb path, unless it is already loaded, in which case use the cached one:
        if imdb_npy_path not in IMDB_CACHE:
            orig_data = np.load(imdb_npy_path, allow_pickle=True)
            IMDB_CACHE[imdb_npy_path] = orig_data
        else:
            orig_data = IMDB_CACHE[imdb_npy_path]

        self.imdb_path = imdb_npy_path
        if indexes:
            # keep only data points of given provider
            selected = [0] + indexes
            data = [orig_data[i] for i in selected]
        else:
            data = orig_data

        self.header = data[0]
        self.imdb = data[1:]

        self.split = split
        self.images_dir = images_dir

        # optionally, allow loading images from hdf5 dataset instead of jpeg:
        if h5_img_path is not None and os.path.exists(h5_img_path):
            # open the h5 file, we'll load the images from it at batch time
            self.use_h5_img = True
            if h5_img_path not in H5_CACHE:
                self.h5_img_file = h5py.File(h5_img_path, 'r')
                H5_CACHE[h5_img_path] = self.h5_img_file
            else:
                self.h5_img_file = H5_CACHE[h5_img_path]
        else:
            self.use_h5_img = False

        self.use_images = kwargs.get('use_images', False)
        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]

        question = record["question"]
        answers = [record['answers'].lower()]
        context = " ".join([word.lower() for word in record['ocr_tokens']])

        if self.get_raw_ocr_data:
            if len(record['ocr_tokens']) == 0:
                words = []
                boxes = np.empty([0, 4])
            else:
                words = [word.lower() for word in record['ocr_tokens']]
                boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])

        if self.use_images:
            if self.use_h5_img:
                # load pre-resized 224x224 images from serialised format directly
                image_names = record['image_name']
                image_arr = self.h5_img_file[image_names][:]
                images = Image.fromarray(image_arr)
            else:
                # original method: decompress the jpegs and resize them inside BEiT
                image_names = os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name']))
                images = Image.open(image_names).convert("RGB")


        qid = record.get('question_id', "{:s}-{:d}".format(record['set_name'], idx))

        sample_info = {
            'question_id': qid,
            'questions': question,
            'contexts': context,
            'answers': answers,
        }


        if self.use_images:
            sample_info['image_names'] = image_names
            sample_info['images'] = images

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes

        return sample_info


def collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch
