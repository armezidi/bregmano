# A Bregman Proximal Viewpoint on Neural Operators

This repository contains the code for the paper "A Bregman Proximal Viewpoint on Neural Operators", accepted at the Forty-Second International Conference on Machine Learning (ICML 2025).

## Repository Structure

```
├── LICENSE                  # MIT license with proper attribution
├── README.md                # This file
├── neuraloperator/          # Modified NeuralOperator library
│   ├── neuralop/            # Core neural operator implementations
│   └── train/               # Training scripts and experiments
├── fourierflow/             # Modified FourierFlow library
├── datasets/                # Datasets for experiments
│   ├── NS_FNO/              # Original Navier-Stokes dataset from FNO paper [link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
│   └── PDEBench/            # PDEBench datasets [link](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
└── wandb/                   # Experiment tracking data
```

## Based on Open Source Libraries

This work builds upon and modifies the following open-source libraries:

### **[NeuralOperator](https://github.com/neuraloperator/neuraloperator)** (MIT License)
- **Base functionality**: FNO, training utilities, and data loaders
- **Our modifications**: 
  - Added Bregman architecture support
  - Zero weight initialization for Bregman architecture

### **[FourierFlow](https://github.com/alasdairtran/fourierflow)** (MIT License)  
- **Base functionality**: Factorized FNO implementation
- **Our modifications**: Custom training to compare with the one used in neuraloperator

### **[Wavelet Neural Operator](https://github.com/TapasTripura/WNO)**
- **Base functionality**: WNO implementation with 1D and 2D support
- **Our modifications**: Implementation of the WNO in the neuraloperator library and added Bregman architecture support

## Datasets

The datasets used in the experiments are the following:
- **FNO Navier-Stokes datasets**: NavierStokes_V1e-3_N5000_T50.zip, NavierStokes_V1e-4_N10000_T30.zip  [Link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
- **PDEBench datasets**: 1D_Advection_Sols_beta0.4.hdf5, 	
1D_Burgers_Sols_Nu0.001.hdf5, 1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5, 	
2D_DarcyFlow_beta0.1_Train.hdf5 [Link](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)

## Installation

- For the **NeuralOperator** library
```bash
cd neuraloperator
pip install -e .
```

- For the **FourierFlow** library, follow the instructions in the [FourierFlow README](fourierflow/README.md).

### Running Experiments
After selecting the parameters in the `neuraloperator/train/train_FNO_pdebench_sweep.py` and `neuraloperator/train/train_WNO_pdebench_sweep.py` files, you can run the experiments by running the following commands:

```bash
# Train FNO with Bregman architecture
python neuraloperator/train/train_FNO_pdebench_sweep.py

# Train WNO with Bregman architecture
python neuraloperator/train/train_WNO_pdebench_sweep.py
```

## Citation

If you use this code, please cite our paper:
```bibtex
@proceedings{mezidi:hal-04584456,
  TITLE = {{A Bregman Proximal Viewpoint on Neural Operators}},
  AUTHOR = {Mezidi, Abdel-Rahim and Patracone, Jordan and Salzo, Saverio and Habrard, Amaury and Pontil, Massimiliano and Emonet, R{\'e}mi and Sebban, Marc},
  URL = {https://inria.hal.science/hal-04584456},
  BOOKTITLE = {{International Conference on Machine Learning}},
  ADDRESS = {Vancouver, Canada},
  YEAR = {2025},
}
```

**Please also cite the original libraries we built upon:**
```bibtex
@misc{kossaifi2024neural,
   title={A Library for Learning Neural Operators},
   author={Jean Kossaifi and Nikola Kovachki and
   Zongyi Li and David Pitt and
   Miguel Liu-Schiaffini and Robert Joseph George and
   Boris Bonev and Kamyar Azizzadenesheli and
   Julius Berner and Anima Anandkumar},
   year={2024},
   eprint={2412.10354},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}

@article{kovachki2021neural,
   author    = {Nikola B. Kovachki and
                  Zongyi Li and
                  Burigede Liu and
                  Kamyar Azizzadenesheli and
                  Kaushik Bhattacharya and
                  Andrew M. Stuart and
                  Anima Anandkumar},
   title     = {Neural Operator: Learning Maps Between Function Spaces},
   journal   = {CoRR},
   volume    = {abs/2108.08481},
   year      = {2021},
}

@inproceedings{tran2023factorized,
  title     = {Factorized Fourier Neural Operators},
  author    = {Alasdair Tran and Alexander Mathews and Lexing Xie and Cheng Soon Ong},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=tmIiMPl4IPa}
}

@article{tripura2023wavelet,
  title={Wavelet Neural Operator for solving parametric partial differential equations in computational mechanics problems},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={404},
  pages={115783},
  year={2023},
  publisher={Elsevier}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The original license notices for the base libraries are preserved in their respective directories (`neuraloperator/LICENSE` and `fourierflow/LICENSE`).
