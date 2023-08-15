# Graph Neural Networks as Gradient Flows

This repository is the official implementation of [Graph Neural Networks as Gradient Flows](https://arxiv.org/abs/2206.10991).

## Requirements

Main dependencies (with python >= 3.7) are: torch==1.8.1 torch-cluster==1.5.9 torch-geometric==2.0.3 torch-scatter==2.0.9 torch-sparse==0.6.12 torch-spline-conv==1.2.1 torchdiffeq==0.2.3 
It is best to install the dependencies in a new conda environment as follows:

```
conda create --name graff python=3.7
conda activate graff
TORCH=1.8.1
CUDA=cu102 or cpu
pip install torch==1.8.1
pip install torchdiffeq==0.2.3 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.0.3
pip install --no-deps deeprobust
```

# Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python run_GNN.py --dataset chameleon --use_best_params --num_splits 1
```

## Cite us
If you found this work useful, please consider citing our paper
```
@misc{digiovanni2023understanding,
      title={Understanding convolution on graphs via energies}, 
      author={Francesco Di Giovanni and James Rowbottom and Benjamin P. Chamberlain and Thomas Markovich and Michael M. Bronstein},
      year={2023},
      eprint={2206.10991},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
