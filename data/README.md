# Downloading Experiment Datasets

Since we use third-party datasets with different licenses/copyrights, 
we only include information on how we downloaded and processed the data.

## FiveThirtyEight's MLB forecasts

Data description: https://fivethirtyeight.com/features/how-our-mlb-predictions-work/ 
  
Download [this `.csv` file](https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv) 
inside this folder as `data/mlb_elo_538.csv`.

Then, load this dataset using
[`cc.data_utils.baseball.read_538_projections`](../comparecast/data_utils/baseball.py).

See also: [`nb_comparecast_baseball.ipynb`](../nb_comparecast_baseball.ipynb).

## Vegas-Odds.com odds on MLB games

Download [these `.xlsx` files](https://sports-statistics.com/sports-data/mlb-historical-odds-scores-datasets/) 
inside this folder as `data/mlb odds 20**.xlsx`.

Then, load this dataset using
[`cc.data_utils.baseball.read_vegas_odds`](../comparecast/data_utils/baseball.py).

See also: [`nb_comparecast_baseball.ipynb`](../nb_comparecast_baseball.ipynb).

## Ensemble weather forecasts & e-values 

Found in: [`eprob/replication_material/precip_fcs/*.csv`](../eprob/replication_material/precip_fcs).

These are downloaded from [the `eprob` repository](https://github.com/AlexanderHenzi/eprob) 
and [processed into `.csv` using R](../eprob/replication_material/convert_precip_fcs_to_csv.R).

To re-run the R code, clone the `eprob` repository somewhere and 
copy the content inside `replication_material` into `$eprob_clone/replication_material`.

Load the dataset using
[`cc.data_utils.weather.read_precip_fcs`](../comparecast/data_utils/weather.py)
(data & forecasts per location/lag)
or
[`cc.data_utils.weather.read_hz_evalues`](../comparecast/data_utils/weather.py)
(data & e-values only).

See also: [`nb_comparecast_weather.ipynb`](../nb_comparecast_weather.ipynb).
