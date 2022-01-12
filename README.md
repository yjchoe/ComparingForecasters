# ComparingForecasters

[![image](https://img.shields.io/pypi/v/comparecast.svg)](https://pypi.org/project/comparecast/)
[![image](https://img.shields.io/pypi/l/comparecast.svg)](https://pypi.org/project/comparecast/)

Code accompanying our paper, [_Comparing Sequential Forecasters_](https://arxiv.org/abs/2110.00115) (arXiv preprint), 
where we derive time-uniform, distribution-free, and non-asymptotic 
confidence sequences (CS), as well as e-processes, 
for comparing forecasts on sequential data. 

## Installation

Requires Python 3.7+.

From `pip`:
```shell
pip install --upgrade pip
pip install --upgrade comparecast
```

From source:
```shell
git clone https://github.com/yjchoe/ComparingForecasters
cd ComparingForecasters

pip install --upgrade pip
pip install -e .
```

## Data Sources

See [`data/README.md`](data/README.md).

## Sample Usage

### Python

```python
import comparecast as cc

# Generate/retrieve synthetic data
data = cc.data_utils.synthetic.get_data("default", size=1000)

# Calculate forecasts
forecasters = ["k29_poly3", "laplace", "always_0.5"]
data = cc.forecast(data, forecasters, out_file="data/test.csv") 
cc.plot_forecasts(data, forecasters, plots_dir="plots/test")

# Compare forecasts using confidence sequences & e-values
lcbs, ucbs, evalues = cc.compare_forecasts(
    data, 
    "k29_poly3", 
    "laplace", 
    scoring_rule="brier", 
    alpha=0.05,
    boundary_type="mixture",
    v_opt=10,
    compute_evalues=True,
)

# Draw a comparison plot
results, axes = cc.plot_comparison(
    data, 
    "k29_poly3", 
    "laplace", 
    scoring_rule="brier", 
    alpha=0.05,
    boundary_type="mixture",
    compare_baselines=("h", "acs"),
    plot_width=True,
    plots_dir="plots/test",
)
```

### Command Line Interface
```shell
# Generate synthetic data and forecasts
python forecast.py -d default -n 1000 -f all \
    -o forecasts/test.csv -p plots/test

# Compare forecasts and plot results
python plot_comparisons.py -d forecasts/test.csv -p k29_poly3 -q laplace \
    --compare-baselines h acs --plot-width -o plots/test
    
# Compare 538 and vegas forecasters
python plot_comparisons.py -d forecasts/mlb_2010_2019.csv \
    -p fivethirtyeight -q vegas \
    --compare-baselines acs --plot-width -o plots/mlb_2010_2019
```

## Experiments

- [**`nb_comparecast_synthetic.ipynb`**](nb_comparecast_synthetic.ipynb): 
  Experiments on synthetic data and forecasts. 
  Includes comparison with other methods (fixed-time CI and asymptotic CS).
  Section 5.1 in our paper.
- [**`nb_comparecast_scoringrules.ipynb`**](nb_comparecast_scoringrules.ipynb): 
  Experiments on synthetic data and forecasts using different scoring rules.
  Section 5.1 in our paper.
- [**`nb_comparecast_baseball.ipynb`**](nb_comparecast_baseball.ipynb): 
  Experiments on Major League Baseball forecasts, 
  leading up to the 2019 World Series.
  Section 5.2 in our paper.
- [**`nb_comparecast_weather.ipynb`**](nb_comparecast_weather.ipynb): 
  Experiments on postprocessing methods for ensemble weather forecasts. 
  Includes e-value comparison with 
  [Henzi & Ziegel (2021)](https://arxiv.org/abs/2103.08402).
  Section 5.3 in our paper.
- [**`nb_comparecast_weather_eda.ipynb`**](nb_comparecast_weather_eda.ipynb): 
  Exploratory plots on the ensemble weather forecast dataset. 
  Section 5.3 in our paper.
- [**`nb_iid_mean.ipynb`**](nb_iid_mean.ipynb): 
  Comparison of uniform boundaries on the mean of IID data.
  Partly reproduces Figure 1 from 
  [Howard et al. (2021)](https://doi.org/10.1214/20-AOS1991).
  Appendix C in our paper.

## License

MIT

## Authors

[YJ Choe](http://yjchoe.github.io/) and 
[Aaditya Ramdas](https://www.stat.cmu.edu/~aramdas/)
