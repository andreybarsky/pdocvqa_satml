This is the participation kit for the [Privacy-Preserving Document VQA competition](https://benchmarks.elsa-ai.eu/?ch=2&com=introduction) organised in the context of SATML 2025. This specific repository is tailored to the Red Team Challenge on membership inference attacks and information reconstruction. For the previous Blue Team Challenge, see [this repo](https://github.com/rubenpt91/PFL-DocVQA-Competition). 

# How to use

- [How to use](#how-to-use)
  - [Set-up environment](#set-up-environment)
  - [Download dataset](#download-dataset)
  - [Train and evaluate](#train-and-evaluate)
  - [Configuration files and input arguments](#configuration-files-and-input-arguments)
    - [Input arguments](#input-arguments)
    - [Datasets configuration files](#datasets-configuration-files)
    - [Models configuration files](#models-configuration-files)
      - [Visual Module](#visual-module)
      - [Training parameters](#training-parameters)
    - [Differential Privacy Parameters](#differential-privacy-parameters)
  - [API calls for red team track 2 competition](#api-calls-for-red-team-track-2-competition)
    - [Use the API](#use-the-api)
    - [Query preparation](#query-preparation)
  - [Monitor experiments](#monitor-experiments)

## Set-up environment

First, clone the repository to your local machine:
```bash
$ git clone https://github.com/andreybarsky/pdocvqa_satml.git
$ cd pdocvqa_satml
```

We recommend using a conda environment to handle the dependencies. If you don't have conda set up, follow the instructions [here](https://docs.anaconda.com/miniconda/) to set up Miniconda, or use your own distribution of choice.

To install all the dependencies, create a new conda environment with the provided yml file:

```bash
$ conda env create -f environment.yml
$ conda activate pdocvqa_satml
```

## Download dataset

1. Download the dataset from the [ELSA Benchmarks Platform](https://benchmarks.elsa-ai.eu/?ch=2&com=downloads).
2. Modify in the dataset configuration file `configs/datasets/PFL-DocVQA-BLUE.yml` the following keys:
    * **imdb_dir**: Path to the imdb directory containing a train and validation data split.
    * **provider_docs**: Path to _centralized_data_points.json_. (for DP training)

   And either:
    * **images_dir**: Path to the dataset images as a directory of jpg files. (used by default)
    * **images_h5_path**: Path to the dataset images as a hdf5 archive. (used with the --use_h5 commandline flag)


## Train and evaluate

To use the framework you only need to call the `train.py` or `eval.py` scripts with the dataset and model you want to use.

The name of the dataset and the model **must** match with the name of the configuration under the `configs/dataset` and `configs/models` respectively. This allows having different configuration files for the same dataset or model. <br>
In addition, to apply or Differential Privacy, you just need to specify ```--use_dp```.

```bash
$ (docvqa_satml) python train.py --dataset PFL-DocVQA-BLUE --model VT5 --use_dp
```

Below, we show a descriptive list of the possible input arguments that can be used.

## Configuration files and input arguments

### Input arguments

| <div style="width:100px">Parameter </div>          | <div style="width:150px">Input param </div> | Required 	  | Description                                                           |
|----------------------------------------------------|---------------------------------------------|-------------|-----------------------------------------------------------------------|
| Model                                              | `-m` `--model`                              | Yes         | Name of the model config file                                         |
| Dataset                                            | `-d` `--dataset`                            | Yes         | Name of the dataset config file                                       |
| Batch size                                         | `-bs`, `--batch-size`                       | No          | Batch size                                                            |
| Initialization seed                                | `--seed`                                    | No          | Initialization seed                                                   |
| Differential Privacy                               | `--use_h5`                                  | No          | Load images from archive file for faster training                     |
| Differential Privacy                               | `--use_dp`                                  | No          | Add Differential Privacy noise                                        |
| Differential Privacy - Sampled providers per Round | `--providers_per_fl_round`                  | No          | Number of groups (providers) sampled in each FL Round when DP is used |
| Differential Privacy - Noise sensitivity           | `--sensitivity`                             | No          | Upper bound of the contribution per group (provider)                  |
| Differential Privacy - Noise multiplier            | `--noise_multiplier`                        | No          | Noise multiplier                                                      |

- Most of these parameters are specified in the configuration files. However, you can overwrite those parameters through the input arguments.

### Datasets configuration files

| Parameter      | Description                               | Values     |
|----------------|-------------------------------------------|------------|
| dataset_name   | Name of the dataset to use.               | PFL-DocVQA |
| imdb_dir       | Path to the numpy annotations file.       | \<Path\>   |
| images_dir     | Path to the images dir.                   | \<Path\>   |
| provider_docs  | Path to the ```data_points.json``` file.  | \<Path\>   |


### Models configuration files

| Parameter           | Description                                                                                                     | Values                                            |
|---------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| model_name          | Name of the dataset to use.                                                                                     | VT5                                               |
| model_weights       | Path to the model weights dir. It can be either local path or huggingface weights id.                           | \<Path\>, \<Huggingface path\>                    |
| max_input_tokens    | Max number of text tokens to input into the model.                                                              | Integer: Usually is 512, 768 or 1024.             |
| save_dir            | Path where the checkpoints and log files will be saved.                                                         | \<Path\>                                          |
| device              | Device to be used                                                                                               | cpu, cuda                                         |
| visual_module       | Visual module parameters <br> Check section                                                                     | [Visual Module](#visual-module)                   |
| training_parameters | Training parameters specified in the model config file.                                                         | [Training parameters](#training-parameters)       |
| dp_parameters       | Differential Privacy parameterstraining parameters are specified in the model config file. <br> Check section   | [DP parameters](#differential-privacy-parameters) |

#### Visual Module

| Parameter     | Description                                                                           | Values                         |
|---------------|---------------------------------------------------------------------------------------|--------------------------------|
| model         | Name of the model used to extract visual features.                                    | ViT, DiT                       |
| model_weights | Path to the model weights dir. It can be either local path or huggingface weights id. | \<Path\>, \<Huggingface path\> |
| finetune      | Whether the visual module should be fine-tuned during training or not.                | Boolean                        |

#### Training parameters

| Parameter           | Description        | Values                 |
|---------------------|--------------------|------------------------|
| lr                  | Learning rate.     | Float (2<sup>-4</sup>) |
| batch_size          | Batch size.        | Integer                |


### Differential Privacy Parameters

| Parameter              | Description                                            | Values           |
|------------------------|--------------------------------------------------------|------------------|
| providers_per_fl_round | Number of groups (providers) sampled in each FL Round. | Integer (50)     |
| sensitivity            | Differential Privacy Noise sensitivity.                | Float   (0.5)    |
| noise_multiplier       | Differential Privacy noise multiplier.                 | Float   (1.182)  |

## API calls for red team track 2 competition
### Use the API

Users can query the model with the provided API code [client.py](./api_red/client.py).  

```bash
python api_red/client.py  --token YOUR_USER_TOKEN --query_path /PATH/TO/query.json --response_save_path /PATH/TO/SAVE/RESPONSE/
```
- `--token`: Your user token obtained during registration as a team in the RED Team Challenge.
- `--query_path`: Path to your query JSON file.
- `--response_save_path`: Directory where the response will be saved


### Query preparation
The query (`query.json`) must be a `JSON` file and should adhere to the following structure:

```json
{
    "numb_requests": n,
    "model": "private" | "non-private",
    "data": [
        {
            "question_ID": "unique_question_id",
            "ocr_tokens": [
                "TOKEN_1",
                "TOKEN_2",
                ...,
                "TOKEN_N"
            ],
            "ocr_normalized_boxes": [
                [
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ],
                ...,
                [
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ]
            ],
            "question": "Your question text here",
            "encoded_image": "Base64_encoded_image_string"
        },
        ...
    ]
}
```
Where:

- `numb_requests`: an Integer representing total number of requests (or questions) that you want the model to predict.

- `model`: a String indicates the model to use for predictions, it can be one of the two options:  
  -  `"private"`: Use the private model.  
  - `"non-private"`: Use the non-private model.  
  
- `ocr_tokens`: Array of Strings that are the words or tokens extracted from the document using OCR (e.g., "TOKEN_1", "TOKEN_2", etc.).  

- `ocr_normalized_boxes`: Array of Arrays, each inner array represents the bounding box coordinates of a token, normalized between 0 and 1 with respect to the image size.  Format: [`x_min`, `y_min`, `x_max`, `y_max`]  

- `question`: String, the question you are asking the model about the document.  

- `encoded_image`: String, A Base64-encoded string representation of the document image that the model will analyze to answer the question.

Example: 
```json
{
    "numb_requests": 1,
    "model": "non-private",
    "data": [
        {
            "question_ID": "1234565454541",
            "ocr_tokens": [
                "TOKEN_1",
                "TOKEN_2",
                "TOKEN_3"
            ],
            "ocr_normalized_boxes": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.7, 0.7, 0.9, 0.8]
            ],
            "question": "What is the date of the invoice?",
            "encoded_image": "iVBORw0KGgoAAAANSUhEUgAACbEAA..."
        }
    ]
}
```

To encode your images into Base64 format, use the provided script [encode_image.py](./api_red/encode_image.py). Run the script with the following command:
```bash
python api_red/encode_image.py --image_dir /PATH/TO/ALL/YOUR/IMAGES/ --output_dir /SAVING/DIR/PATH/
```
The images will be saved as `JSON` files with the following structure:
```json
{
    "encoded_image": "iVBORw0KGgoAAAANSUhEUgAACbEAA..."
}
```
## Monitor experiments

By default, the framework will log all the training and evaluation process in [Weights and Biases (wandb)](https://wandb.ai/home). <br>

<div style="text-align: justify;">
The first time you run the framework, wandb will ask you to provide your wandb account information.
You can decide either to create a new account, provide a <a href="https://wandb.ai/authorize">authorization token</a> from an already existing account, or do not visualize the logging process.
The two first options are straightforward and wandb should properly guide you.
In the case you don't want to visualize the results you might get an error when running the experiments.
To prevent this, you need to disable wandb by typing:
</div>

```bash
$ (pdocvqa_satml) export WANDB_MODE="offline"
```
