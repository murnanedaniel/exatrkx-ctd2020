# CTD2020 ExatrkX

Each `stage` of the pipeline can be executed separately by running `python [path/to/config]` followed by one of the stages:

<pre>
      <a href="MetricLearning/src/preprocess_with_dir" title="Preprocessing function">preprocess</a>             <a href="MetricLearning/src/metric_learning_adjacent" title="Doublet building function">build_doublets</a>           <a href="GraphLearning/src/" title="Triplet building function">build_triplets</a>                 <a href="Seeding/src" title="Seeding function function">seed</a>          <a href="Labelling" title="Labelling function">label</a>
</pre>

![](docs/pipeline.png)

## Setting Up Environment

These instructions assume you are running [miniconda](https://docs.conda.io/en/latest/miniconda.html) on a linux system.
1. Create an exatrkx conda environment
```bash
source [path_to_miniconda]/bin/activate
conda create --name exatrkx python=3.7
conda activate exatrkx
```
2. nb_conda_kernels allows to pick the exatrkx conda kernel from a traditional jupyter notebook. To install it (in your base environment or in the exatrkx environment)
```
conda install nb_conda_kernels
```

## Installation

To leverage all available resources in your system, there are two versions of some functions in this library - one compatible with CUDA GPU acceleration, and one for CPU operations. To install correctly, first run
```
nvcc --version
```
to get your CUDA version. If it returns an error, you do not have GPU enabled, then set the environment variable to `export CUDA=cpu`. Otherwise set your variable to `export CUDA=cuXXX`, where cuXXX is cu92, cu101 or cu102 for CUDA versions 9.2, 10.1, or 10.2. Setting this variable correctly is essential to having all libraries aligned correctly. Then install PyTorch requirements
```
pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```
Finally run the setup to install all other packages

```
pip install -e .
```

## Directory Structure

An entire `classify` or `train` pipeline can be run from the root directory using the `pipeline.py` script. Stages of these pipelines will produce either `classify/` or `train/` and `artifact/` data.

## Example Run

To run the full pipeline to build seeds from TrackML data contained in `/path/to/trackml-data` (which can be downloaded from [Kaggle](https://www.kaggle.com/c/trackml-particle-identification), we first need model files that will be used for the learned embedding and GNN classifications. These four folders (embedding, filter, doublet GNN & triplet GNN) can be downloaded from [NEED TO CONFIRM DOWNLOAD LOCATION](www.google.com).

Assuming these have been downloaded and stored in `/path/to/artifacts`, we alter a config file to point to these locations, as well as save paths for intermediate data and final seed data. To differentiate this run, we give it a name. An example config file `seed_example.yaml` with the experiment name "My_First_Run" is in [the seed config folder](Seeding/src/configs). We run
```
python pipeline.py Seeding/config/seed_example.yaml seed
```

This will produce a full set of event files in the seed folder (note that seeding is the default, so the `seed` specification is actually unnecessary in this case).

## Other Use Cases

### Labelling

Similar to the seeding case, once model artifacts are available (through download or by running the `train` pipeline), run
```
python pipeline.py Label/config/label_example.yaml label
```
to produce sets of labels in `data/storage/path/classify/labels`.

### Training

The pipeline is set up to train the models required for the building stages. Just as in the building pipeline, one can run up to any stage of training, where stages are given as 
```
preprocess --> train_embedding --> train_filter --> train_doublets --> train_triplets
```
Each of the `train_X` stages will produce model artifacts in the folder given by the experiment name specified in the config file. Once the full training pipeline is complete, one can then run any seeding, labelling or classification stages detailed above.

N.B. Each stage of training requires intermediate data files to be created (training data must be assembled into graphs, etc. to be trained), which will be deleted after it is no longer needed. You can turn off this deletion in the training config, but it should not be needed

## Further Configuration and Experimentation
