# SensitivitySurrogacy

This package implements methods for sensitivity analysis and partial identification of long-term treatment effects using surrogate outcomes. The methods evaluate the robustness of the surrogate index approach by relaxing the **surrogacy assumption**, providing worst-case bounds and sensitivity analyses that quantify how estimated treatment effects change as the assumption is violated. The degree of violation is parameterized through a copula parameter governing the dependence between treatment and the long-term outcome conditional on baseline covariates and surrogates.

Both R and Python implementations are available.

The package includes two main procedures:

- `longterm_copula()`: estimates long-term treatment effects for a copula-based sensitivity analysis
- `longterm_partial_id()`: computes worst-case (partial identification) bounds for long-term treatment effects

For methodological details, see our paper:

[https://arxiv.org/pdf/2603.00580](https://arxiv.org/pdf/2603.00580)

## R Package

### Installation

You can install the development version from GitHub with:

```r
# install.packages("remotes")
remotes::install_github("yqi3/SensitivitySurrogacy")
```

## Python Package

### Installation

```
git clone https://github.com/yqi3/SensitivitySurrogacy.git
cd SensitivitySurrogacy/python
pip install .
