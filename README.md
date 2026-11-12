# From Data to Forecasts: Modern Time-Series Forecasting with Deep Learning

**DevFest Sherbrooke 2026** · Farah Hammami

Forecasting Ontario's hourly electricity demand 24 hours ahead with TensorFlow/Keras and Google's TimesFM, and checking whether deep learning really beats simple baselines.

| Notebook | What it does | Open |
|---|---|---|
| **Demo** | The 8-minute walkthrough shown in the talk. Loads the saved results, runs in seconds. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarahHammamii/devFest_Sherbrooke_Time_Series/blob/main/notebooks/DevFest2026_Forecasting_Demo.ipynb) |
| **Full pipeline** | Data download, exploration, training, TimesFM, evaluation. Fully reproducible (~10 min on a T4 GPU). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarahHammamii/devFest_Sherbrooke_Time_Series/blob/main/notebooks/DevFest2026_Forecasting_Full.ipynb) |

Slides: [`slides/From_Data_to_Forecasts_DevFest2026.pdf`](slides/From_Data_to_Forecasts_DevFest2026.pdf)

## The experiment

- **Data:** IESO hourly Ontario demand (MW), January 2020 to November 2026, 60 169 observations.
- **Task:** every day at midnight, forecast the next 24 hours from the previous 168 hours plus calendar features.
- **Split (chronological):** train until June 2025, validation July–December 2025, test on 2026.
- **Models:** seasonal baselines, a one-layer linear Keras model, an LSTM, a quantile LSTM (P10/P50/P90, pinball loss) and TimesFM 2.5 used zero-shot.

## Results on 2026

| Model | MAE (MW) | RMSE (MW) | MAE vs "same hour last week" |
|---|---:|---:|---:|
| Quantile LSTM (P50) | 530 | 729 | −54 % |
| Linear (Keras) | 550 | 768 | −52 % |
| LSTM (Keras) | 558 | 744 | −52 % |
| TimesFM 2.5 (zero-shot) | 586 | 840 | −49 % |
| Same hour yesterday | 811 | 1069 | −30 % |
| Same hour last week | 1150 | 1546 | ref. |

P10–P90 interval coverage (nominal 80 %): quantile LSTM 72.8 %, TimesFM 76.3 %.

Takeaways: every learned model halves the baseline error; a linear model matches the LSTM; TimesFM gets within 5 % of the trained LSTM with no training; both prediction intervals are slightly overconfident.

## Repository

```
notebooks/   demo and full notebooks (with outputs)
artifacts/   data and outputs used by the demo notebook
figures/     charts produced by the full notebook
slides/      talk slides (PDF)
```

To reproduce everything, open the full notebook, select a T4 GPU runtime and run all cells. It regenerates `artifacts/` and `figures/`.

## References

- Nie et al., *A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST), ICLR 2023
- Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023
- Das et al., *A decoder-only foundation model for time-series forecasting* (TimesFM), ICML 2024 · [github.com/google-research/timesfm](https://github.com/google-research/timesfm)
- IESO public reports: [reports-public.ieso.ca/public/Demand](https://reports-public.ieso.ca/public/Demand/)

## License

Code: MIT. Data: Independent Electricity System Operator (IESO), public reports, subject to IESO terms of use. TimesFM 2.5 weights: Apache-2.0.
