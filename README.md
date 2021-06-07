# Neural Relation Understanding 
Naru is a suite of neural cardinality estimators for tabular data.

![GitHub](https://img.shields.io/github/license/naru-project/naru.svg?color=green)

This repo contains the code for the VLDB'20 paper, [_Deep Unsupervised Cardinality Estimation_](#reference).  

Main modules:

- [`common.py`](./common.py): a lightweight `pandas`-based library to load/analyze/represent tables 
- several deep autoregressive model [architectures](#model-architectures) 
- [`ProgressiveSampling`](./estimators.py): approximate inference algorithms for deep autoregressive models
- a generator for high-dimensional SQL queries
- training/evaluation scripts 

## Quick start

To set up a conda environment, run:

```bash
conda env create -f environment.yml
```

Run the following to test on a tiny 100-row dataset:
```bash
source activate naru

# Trains a ResMADE on dataset 'dmv-tiny'.
# This will create a checkpoint with path 'models/dmv-tiny-<model spec>.pt'.
python train_model.py --epochs=100 --residual 

# Use the trained model as a cardinality estimator.
# --glob supports evaluating a set of checkpoints at once; here, there will only be one match.
python eval_model.py --glob='dmv-tiny*.pt' --residual
```

## Model architectures

Naru currently implements three state-of-the-art autoregressive architectures:

1. **[MADE](./made.py)**: a highly efficient masked MLP, introduced in [Masked Autoencoder for Distribution Estimation (ICML'15)](https://arxiv.org/abs/1502.03509). 
2. **[ResMADE](./made.py)**: MADE with residual connections, introduced in [Autoregressive Energy Machines (ICML'19)](http://proceedings.mlr.press/v97/durkan19a/durkan19a.pdf). 
3. **[Transformer](./transformer.py)**: an autoregressive [Transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), the architecture powering several recent breakthroughs in natural language processing (e.g., BERT, GPT-2, XLNet).

In principle, Naru's inference algorithms can interface with any autoregressive model and turn them into cardinality estimators.

## Datasets

**DMV**.  The DMV dataset is publically available at [catalog.data.gov](https://catalog.data.gov/dataset/vehicle-snowmobile-and-boat-registrations).  The data is continuously updated.  Our frozen snapshot (~11.6M tuples) can be downloaded by running
```bash
bash ./download_dmv.sh
```
Specify `--dataset=dmv` when launching the training/evaluation scripts.

### Registering custom datasets

A user can point a Naru model to her own datasets in a few steps.

First, put a CSV file under `datasets/`.  Second, define in [datasets.py](./datasets.py) a `LoadMyDataset()` function:
```python
def LoadMyDataset(filepath):   
    # Make sure that this loads data correctly.  
    df = pd.read_csv(filepath, **kwargs)  
    return CsvTable('Name of Dataset', df, cols=df.columns)
```
Last, call this function in the appropriate places inside the train/evaluation scripts.  Search for current usage of `args.dataset` in those files and extend accordingly.

## Running experiments
Run `python train_model.py --help` to see a list of tunable knobs.  We recommend at least setting `--residual --direct-io --column-masking`.  (In terms of learning efficiency, ResMADE learns faster than MADE, and `--direct-io` also helps.  Architecture:  Transformer can achieve lower negative log-likelihoods so it fits complex datasets better albeit being more expensive.)

When running evaluation (`eval_model.py`), include the same set of architecture flags to make sure checkpoint loading is correct.

Examples:
```bash
# Use a small 256x5 ResMADE model, with column masking.
python train_model.py --num-gpus=1 --dataset=dmv --epochs=20 --warmups=8000 --bs=2048 \
    --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking

# Evaluate.  To enable estimators other than Naru, see section below.
python eval_model.py --dataset=dmv --glob='<ckpt from above>' --num-queries=2000 \
    --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking
    
# Alternative: larger MADE model reported in paper.
python train_model.py --num-gpus=1 --dataset=dmv --epochs=100 --warmups=12000 --bs=2048 \
    --layers=0 --direct-io --column-masking --input-encoding=binary --output-encoding=one_hot

# Alternative: use a Transformer.
python train_model.py --num-gpus=1 --dataset=dmv --epochs=20 --warmup=20000 --bs=1024 \
    --blocks=4 --dmodel=64 --dff=256 --heads=4 --column-masking
```

## Baseline cardinality estimators
We also include a set of baseline cardinality estimators known in the database literature:

* Naru (`--glob` to find trained checkpoints)
* Sampling (`--run-sampling`)
* Bayes nets (`--run-bn`)
* MaxDiff n-dimensional histogram (`--run-maxdiff`)
* Postgres (see `estimators.Postgres`)

Example: to run experiments using trained Naru model(s) and a Sampler:
```bash
python eval_model.py --dataset=dmv --num-queries=2000 --glob='dmv*.pt' --run-sampling
```
Parameters controling these estimators can be adjusted inside [`eval_model.py`](https://github.com/concretevitamin/naru/blob/master/eval_model.py#L519).

## Contributors
This repo was written by: [Amog Kamsetty](https://github.com/amogkam), [Chenggang Wu](https://github.com/cw75), [Eric Liang](https://github.com/ericl), [Zongheng Yang](https://github.com/concretevitamin).

## Reference

If you find this repository useful in your work, please cite [our VLDB'20 paper](http://www.vldb.org/pvldb/vol13/p279-yang.pdf):

```
@inproceedings{naru,
  title={Deep Unsupervised Cardinality Estimation},
  author={Yang, Zongheng and Liang, Eric and Kamsetty, Amog and Wu, Chenggang and Duan, Yan and Chen, Xi and Abbeel, Pieter and Hellerstein, Joseph M and Krishnan, Sanjay and Stoica, Ion},
  journal={Proceedings of the VLDB Endowment},
  volume={13},
  number={3},
  pages={279--292},
  year={2019},
  publisher={VLDB Endowment}
}
```
