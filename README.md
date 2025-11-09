## **A Fair Empirical Risk Minimization with Generalized Entropy**

This repository contains an official implementation of the paper:
https://arxiv.org/abs/2202.11966

---

#### **REQUIREMENTS**

- Python 3.8 or higher
- g++ supporting C++11

#### **USAGE**

1.  Install dependencies

        python -m pip install .

2.  Run experiments

        # plot convergence curves
        python main.py --study_type convergence --dataset_name adult --metrics t:ge_bar_trace --metrics t:err_bar_trace --lambda_max 20.0 --nu 0.01 --alpha 0.0 --gamma 0.04 --c 8.0 --a 5.0

        # plot I_alpha and error by varying gamma values
        python main.py --study_type varying_gamma --dataset_name adult --metrics v:ge[0] --metrics v:err[0] --lambda_max 20.0 --nu 0.01 --alpha 0.0 --gamma "np.linspace(0.02, 0.07, 20)" --c 8.0 --a 5.0

3.  Then the outputs will be saved in `./output/` directory.

Type `python main.py -h` for more CLI options.

Check [`example.ipynb`](./example.ipynb) for API usage.

#### **NOTE**

- Running time varies depending on the parameters. Above two CLI examples in _USAGE_ section will take about 30 seconds each on general computing environment.

