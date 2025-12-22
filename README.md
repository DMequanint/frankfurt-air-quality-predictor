# frankfurt-air-quality-predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Frankfurt PM2.5 forecasting with WHO compliance using OpenAQ API

## Project Organization

```
frankfurt-air-quality-predictor/                    # PROJECT ROOT
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── .gitignore                                      # CREATE THIS
│
├── data/
│   ├── raw/
│   └── processed/                                 # Keep your CSVs temporarily
│
├── models/                                        # Keep your .pkl files temporarily
│
├── reports/
│   └── figures/                                   # Keep your PNG plots
│
└── frankfurt_air_quality_predictor/               # ALL SOURCE CODE
    ├── __init__.py
    ├── config.py
    ├── dataset.py                                 # Data pipeline
    ├── features.py                                # Feature engineering
    ├── modeling/
    │   ├── __init__.py
    │   ├── train.py                               # Dual XGBoost
    │   └── predict.py                             # Live predictions
    └── plots.py                                   # Visualizations

```

--------

