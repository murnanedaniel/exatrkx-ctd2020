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

## Installing Requirements

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

## Example Run

To run the full pipeline to build seeds from TrackML data contained in `/path/to/trackml-data`, we first need model files that will be used for the learned embedding and GNN classifications. Assuming these have been downloaded and stored in `/path/to/artifacts`, we alter a config file to point to these locations, as well as save paths for intermediate data and final seed data. An example config file is in [the seed config folder](Seeding/src/configs). We run
```
python pipeline.py path/to/config/seed.yaml seed
```

This will produce a full set of event files in the seed folder.

## Use Cases

## Further Configuration and Experimentation
