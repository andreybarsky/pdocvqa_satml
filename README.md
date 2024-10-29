This is the participation kit for the [Privacy-Preserving Document VQA competition](https://benchmarks.elsa-ai.eu/?ch=2&com=introduction) organised in the context of SATML 2025. This specific repository is tailored to the Red Team Challenge on membership inference attacks and information reconstruction. For the previous Blue Team Challenge, see [this repo](https://github.com/rubenpt91/PFL-DocVQA-Competition). 

# How to use

1. [Set-up environment](#set-up-environment)
2. [Download Dataset](#download-dataset)
3. [Download Pretrained Weights](#download-pretrained_weights)
4. [Train and evaluate](#train-and-evaluate)
5. [Configuration files and input arguments](#configuration-files-and-input-arguments)
   1. [Input Arguments](#input-arguments)
   2. [Datasets configuration files](#datasets-configuration-files)
   3. [Models configuration files](#models-configuration-files)
      1. [Visual Module](#visual-module)
      2. [Training parameters](#training-parameters)
      3. [Federated Learning parameters](#federated-learning-parameters)
      4. [Differential Privacy Parameters](#differential-privacy-parameters)
6. [Monitor experiments](#monitor-experiments)
7. [Project Structure](#project-structure)

## Set-up environment

First, clone the repository to your local machine:
```bash
$ git clone https://github.com/andreybarsky/pdocvqa_satml.git
$ cd pdocvqa_satml
```

To install all the dependencies, you need to create a new conda environment with the provided yml file:

```bash
$ conda env create -f environment.yml
$ conda activate pdocvqa_satml
```


## Download dataset

1. Download the dataset from the [ELSA Benchmarks Platform](https://benchmarks.elsa-ai.eu/?ch=2&com=downloads).
2. Modify in the dataset configuration file `configs/datasets/PFL-DocVQA.yml` the following keys:
    * **imdb_dir**: Path to the imdb directory with all train and validation clients.
    * **provider_docs**: Path to _centralized_data_points.json_. (for DP training)
   And either:
    * **images_dir**: Path to the dataset images as a directory of jpg files. (used by default)
    * **images_h5_path**: Path to the dataset images as a hdf5 archive. (used with the --use_h5 commandline flag)

### (Optional) Download pretrained weights

1. Download the [pretrained weights](https://datasets.cvc.uab.es/elsa/PFL-DocVQA/vt5_mp-docvqa.ckpt.zip) on SP-DocVQA.
2. Unzip the weights `unzip vt5_mp-docvqa.ckpt.zip`.
3. Change the `model_weights` path in `configs/models/VT5.yml` to point to your local ''_.ckpt_'' path.

## Train and evaluate

To use the framework you only need to call the `train.py` or `eval.py` scripts with the dataset and model you want to use.

The name of the dataset and the model **must** match with the name of the configuration under the `configs/dataset` and `configs/models` respectively. This allows having different configuration files for the same dataset or model. <br>
In addition, to apply or Differential Privacy, you just need to specify ```--use_dp```.

```bash
$ (docvqa_satml) python train.py --dataset PFL-DocVQA --model VT5 --use_dp
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
