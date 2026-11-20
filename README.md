# Valencia compound drought–heat forecasting

Retrospective seven-day drought–heat forecast evaluation in southern Valencia using 27 native ERA5 cells. Pooled prediction sets achieved 94.64% overall test coverage and included the event on 0 of 59 positive cell-days. Class-conditional calibration included the event on 43 of 59. Persistence outperformed the temporal convolutional network on average precision. The pilot has no operational forecast validation.

[Portfolio case study](https://www.daudee.com/work/valencia-drought-heat)

## Inspect the published results

- [Conformal coverage](metrics/conformal_coverage.csv) records event-conditional and aggregate coverage.
- [Discrimination](metrics/discrimination.csv) compares persistence and learned models.
- [Split support](metrics/split_support.csv) reports the positive counts across evaluation periods.

## Run

Use Python 3.12 in a clean virtual environment. The [reproduction guide](REPRODUCE.md) describes source acquisition and generated outputs. A full run downloads the ERA5 series and boundary geometry, then rebuilds the climate cube, checkpoints and metrics. The exact historical run depends on the original provider responses.

```sh
python -m pip install -r requirements-lock.txt
python src/acquire.py
python src/prepare.py
python src/train.py
python src/evaluate.py
python src/deliver.py
python -m pytest -q
```

## Data and attribution

ERA5 via Open-Meteo and Eurostat GISCO administrative boundaries (© EuroGeographics for the administrative boundaries). The repository contains aggregate metrics. Source geometry, reanalysis responses and trained checkpoints are acquired or regenerated separately.
