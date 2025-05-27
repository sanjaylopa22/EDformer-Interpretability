# Interpreting Multi-Horizon Time Series Deep Learning Models

Interpreting the model's behavior is important in understanding decision-making in practice. However, explaining complex time series forecasting models faces challenges due to temporal dependencies between subsequent time steps and the varying importance of input features over time. Many time series forecasting models use input context with a look-back window for better prediction performance. However, the existing studies (1) do not consider the temporal dependencies among the feature vectors in the input window and (2) separately consider the time dimension that the feature dimension when calculating the importance scores. In this work, we propose a novel **Windowed Temporal Saliency Rescaling** method to address these issues. 

## Core Libraries
The following libraries are used as a core in this framework.

### [Captum](https://captum.ai/docs/introduction)
(“comprehension” in Latin) is an open source library for model interpretability built on PyTorch.

### [Time Interpret (tint)](https://josephenguehard.github.io/time_interpret/build/html/index.html)

Expands the Captum library with a specific focus on time-series. It includes various interpretability methods specifically designed to handle time series data.

### [Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library)

TSlib is an open-source library for deep learning researchers, especially deep time series analysis.


## Interpretation Methods

The following local intepretation methods are supported at present:
<details>
1. *Feature Ablation* [[2017]](https://arxiv.org/abs/1705.08498)
2. *Dyna Mask* [[ICML 2021]](https://arxiv.org/abs/2106.05303)
3. *Extremal Mask* [[ICML 2023]](https://proceedings.mlr.press/v202/enguehard23a/enguehard23a.pdf)
4. *Feature Permutation* [[Molnar 2020]](https://christophm.github.io/interpretable-ml-book/)
5. *Augmented Feature Occlusion* [[NeurIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/08fa43588c2571ade19bc0fa5936e028-Paper.pdf)
6. *Gradient Shap* [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
7. *Integreated Gradients* [[ICML 2017]](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
8. *WinIT* [[ICLR 2023 poster]](https://openreview.net/forum?id=C0q9oBc3n4)
9.  *TSR* [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf)
10. *WinTSR* - proposed new method
</details>

## Time Series Models 
This repository currently supports the following models:

<details>

- [x] **EDformer** - A lightweight reverse embedded transformer model
  
</details>

## Train & Test

Use the [run.py](/run.py) script to train and test the time series models. Check the [scripts](/scripts/) and [slurm](/slurm/) folder to see sample scripts. Make sure you have the datasets downloaded in the `dataset` folder following the `Datasets` section. Following is a sample code to train the electricity dataset using the DLinear model. To test an already trained model, just remove the `--train` parameter.

python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model EDformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --dec_out 1 \
   --itr_no 1

# feature_ablation occlusion augmented_occlusion feature_permutation
# deep_lift gradient_shap integrated_gradients -- only for transformer models
python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation gradient_shap\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model EDformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --dec_out 1 \
   --itr_no 1

## Interpret

Use the [interpret.py](/interpret.py) script to interpret a trained model. Check the [scripts](/scripts/) and [slurm](/slurm/) folder to see more sample scripts. Following is a sample code to interpret the `iTransformer` model trained on the electricity dataset using using some of the interpretation methods. This evaluates the 1st iteration among the default 3 in the result folder.

```
python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation augmented_occlusion feature_permutation integrated_gradients gradient_shap wtsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model EDformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --itr_no 1
```

## Datasets

The datasets are available at this [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) in the long-term-forecast folder. Download and keep them in the `dataset` folder here. Only `mimic-iii` dataset is private and hence must be approved to get access from [PhysioNet](https://mimic.mit.edu/docs/gettingstarted/).

### Electricity

The electricity dataset [^1] was collected in 15-minute intervals from 2011 to 2014. We select the records from 2012 to 2014 since many
zero values exist in 2011. The processed dataset contains
the hourly electricity consumption of 321 clients. We use
’MT 321’ as the target, and the train/val/test is 12/2/2 months. We aggregated it to 1h intervals following prior works.  

### Traffic

This dataset [^2] records the road occupancy rates from different sensors on San Francisco freeways.


## Reproduce

The module was developed using python 3.10.

### Option 1. Use Container

[Dockerfile](/Dockerfile) contains the docker buidling definitions. You can build the container using 
```
docker build -t timeseries
```
This creates a docker container with name tag timeseries. The run our scripts inside the container. To create a `Singularity` container use the following [definition file](/singularity.def).
```
sudo singularity build timeseries.sif singularity.def
```
This will create a singularity container with name `timeseries.sif`. Note that, this requires `sudo` privilege.

### Option 2. Use Virtual Environment
First create a virtual environment with the required libraries. For example, to create an venv named `ml`, you can either use the `Anaconda` library or your locally installed `python`. An example code using Anaconda,

```
conda create -n ml python=3.10
conda activate ml
```
This will activate the venv `ml`. Install the required libraries,

```bash
python3 -m pip install -r requirements.txt
```

If you want to run code on your GPU, you need to have CUDA installed. Check if you already have CUDA installed. 

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} backend')
```

If this fails to detect your GPU, install CUDA using,
```bash
pip install torch==2.2 --index-url https://download.pytorch.org/whl/cu118
```

## References
<!-- https://docs.github.com/en/enterprise-cloud@latest/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#footnotes -->

[^1]: https://arxiv.org/abs/2412.12227

# Citation
If you find this repo useful, please cite our paper.

@article{chakraborty2024edformer, title={EDformer: Embedded Decomposition Transformer for Interpretable Multivariate Time Series Predictions}, author={Chakraborty, Sanjay and Delibasoglu, Ibrahim and Heintz, Fredrik}, journal={arXiv preprint arXiv:2412.12227}, year={2024} }

# Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:

Sanjay Chakraborty (Postdoc, sanjay.chakraborty@liu.se)
