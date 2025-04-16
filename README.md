# IOHGNBG

**IOHGNBG** also enables the integration of GNBG problems into IOHprofiler standards. It is provided as a library for use in the [GNBG Competition 2025](https://gecco-2025.sigevo.org/Competition?itemId=2781), with additional details available on the [competition website](https://dsmlossf.github.io/GNBG-Competition-2025/).

As part of the IOHprofiler ecosystem, IOHGNBG is actively developed, with ongoing updates to its features and interfaces.


## Installation

The minimum supported Python version is 3.10. Install IOHGNBG via pip and git:

```bash
pip install iohgnbg
```

## Basic Usage
The following example shows how to get a benchmark GNBG problem (2025 GECCO competition):
```python
import iohgnbg
import ioh
import os
gnbg_problem = iohgnbg.get_problem(1)

print(gnbg_problem.meta_data)

logger = ioh.logger.Analyzer(
    root=os.getcwd(),  # Current working directory
    folder_name="AttachedLogger",  # Folder to store logs
    algorithm_name="None",  # Name of the algorithm (can be customized)
)

# Attach the logger to the created clustering problem
gnbg_problem.attach_logger(logger)
```



## Tutorials

Explore the following Jupyter notebooks for step-by-step tutorials on using **IOHGNBG**:
1. [Random Search Tutorial ](https://github.com/IOHprofiler/IOHGNBG/blob/main/tutorials/random_search_tutorial.ipynb): Learn how to define a GNBG optimization problems, solve with Random Search and integrate with IOHinspector.

## License

This project is licensed under a standard BSD-3 clause License. See the LICENSE file for details.

## Acknowledgments
We acknowledge the foundational work on the GNBG generator proposed in the paper:

* **"GNBG: A generalized and configurable benchmark generator for continuous numerical optimization"** by Danial Yazdani, Mohammad Nabi Omidvar, Delaram Yazdani, Kalyanmoy Deb, and Amir H. Gandomi ([arXiv:2312.07083](https://arxiv.org/abs/2312.07083)).

The basic instances provided are from the [GECCO Competition 2025](https://github.com/rohitsalgotra/GNBG-II). We also offer participants the possibility to use IOHprofiler to log their experiments.

