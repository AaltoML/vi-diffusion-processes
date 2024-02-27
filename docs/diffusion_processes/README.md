# Variational Gaussian Process Diffusion Processes

This directory contains the experiments included in the publication:

* P. Verma, V. Adam, and A. Solin, (2024). **Variational Gaussian Process Diffusion Processes**. In *Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) 2024*. [arXiv](https://arxiv.org/abs/2306.02066)

## Requirements

To install additional packages required for the experiments in this directory, run the following command:

```shell
pip install -r dp_requirements.txt
```

## Hydra configurations
All the experiments are configured using the [Hydra](https://hydra.cc/) configuration system. 
The configuration files are available in the `configs/` directory from where the hyperparameters for each experiments can be set.
If you want to run the experiments with different hyperparameters, you can modify the configuration files or pass the hyperparameters as command line arguments. 

## Synthetic  diffusion processes experiments
This section discuss the experiments to re-create Table 1-2, Fig. 1-4, and Fig. A8-A11 of the paper.

### Generate the data
The data can be generated using the script `generate_data.py`. A sample command for generating data for a linear (Ornstein--Uhlenbeck) process is:
```shell
python generate_data.py -d=0.5 -q=1.0 -t0=0.0 -t1=100. -x0=1 -dt=0.01 -n=50 --o="data/linear" -s=15 -sde="ou" -k=5
```
More arguments can be seen by the command `python generate_data.py --help`

**Note:** The data used for the experiments in the paper can be found in `docs/diffusion_processes/data.zip`.

### Inference experiments
We can run the experiments when only inference is performed using the following sample commands:

**VDP (Archambeau et al., 2007)**
```shell
python vi_markov_gp.py --config-name="vi_linear_process.yaml" data_path="data/linear/15/0.npz" prior_sde="ou" prior_sde.decay=1.2 dt=0.01 trainer.warmup_x0_itr=5 trainer.q_lr=0.1 trainer.x0_lr=0.1
```
The above command runs the VDP model for the linear DP (Ornstein-Uhlenbeck). 

**CVI-DP (Verma et al., 2024)**
```shell
python cvi_dp.py --config-name="cvi_linear_process.yaml" data_path="data/linear/15/0.npz" prior_sde="ou" prior_sde.decay=1.2 dt=0.01
```
The above command runs the CVI-DP model for the linear DP (Ornstein-Uhlenbeck). 

To run the CVI-DP model for the non-linear DP, use the following command:
```shell
python cvi_dp.py --config-name="cvi_non_linear_process.yaml" data_path="data/dw/3/0.npz" prior_sde="dw" dt=0.01 trainer.max_itr_sites_optim=20 trainer.girsanov_sites_lr=0.5 trainer.data_sites_lr=0.5 prior_sde.c=1.0
```

**GPR model**
```shell
python gpr_linear.py data_path="data/linear/15/0.npz" decay=1.2
```
The above command runs the GPR model for the linear DP (Ornstein-Uhlenbeck).

To run the GPR model for the non-linear DP, use the following command:
```shell
python gpr_non_linear.py data_path="data/dw/3/0.npz"
```

### Learning experiments
We can run the experiments when both inference and learning are performed using the following sample commands:

**VDP (Archambeau et al., 2007)**
```shell
python vi_markov_gp.py --config-name="vi_linear_process.yaml" data_path="data/linear/15/0.npz" prior_sde="ou" prior_sde.decay=2.5 dt=0.01 trainer.learn_prior_sde=true trainer.warmup_x0_itr=0 trainer.optimize_prior_initial_state=true trainer.q_lr=0.1 trainer.x0_lr=0.1 trainer.prior_sde_lr=0.01
```
The above command runs the VDP model for the linear DP (Ornstein-Uhlenbeck).

**CVI-DP (Verma et al., 2024)**
```shell
python cvi_dp.py --config-name="cvi_linear_process.yaml" data_path="data/linear/15/0.npz" prior_sde="ou" prior_sde.decay=2.5 dt=0.01 trainer.learn_prior_sde=true trainer.learning_max_itr=1000 trainer.max_itr=10
```
The above command runs the CVI-DP model for the linear DP (Ornstein-Uhlenbeck).

**GPR model**
```shell
python gpr_linear.py data_path="data/linear/15/0.npz" decay=2.5 optimize=true
```
The above command runs the GPR model for the linear DP (Ornstein-Uhlenbeck).

## Real world experiment
This section discusses the experiments to re-create Fig. 6,7, and A12 of the paper.

### Finance data experiment
The Apple Inc. share price experiment can be performed using the following sample commands:

**VDP (Archambeau et al., 2007)**
```shell
python vi_markov_gp.py --config-name=vi_apple_stock_process.yaml data_path="data/apple_stock/0_data.npz" dt=0.001
```

**CVI-DP (Verma et al., 2024)**
```shell
python cvi_dp.py --config-name=cvi_apple_stock_process.yaml data_path="data/apple_stock/0_data.npz"
```

**GPR**
```shell
cd stock/
python gpr_stock.py data_path="../data/apple_stock/0_data.npz" constant_kernel_variance=0.25 linear_kernel_variance=0.25
```

**SGPR**
```shell
cd stock/
python sgpr_stock.py data_path="../data/apple_stock/0_data.npz"
```

### Vehicle tracking experiment
The vehcile tracking experiment can be performed using the following sample comm+ands:

**VDP (Archambeau et al., 2007)**
```shell
python vi_markov_gp.py --config-name=vi_gps_process.yaml data_path="data/gps/0_data.npz"  dataloader.train_dim=0
```
**CVI-DP (Verma et al., 2024)**
```shell
python cvi_dp.py --config-name=cvi_gps_process.yaml data_path="data/gps/0_data.npz" dataloader.train_dim=0
```
## Comparison with NeuralSDE method
The comparison experiment with NeuralSDE is implemented using the [torchsde](https://github.com/google-research/torchsde) library. 
The code is available in the directory `docs/difussion_process/neuralsde` and a sample experiment can be run by the following command:
```shell
cd neuralsde
python main.py --data_dir="../data/dw/3/0.npz"
```
