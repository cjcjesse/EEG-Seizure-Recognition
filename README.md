# EEG Seizure Recognition

Exploring time-series and time-series derivative approaches to EEG seizure recognition.

## Project Organization

```
├── LICENSE            
├── Makefile           
├── README.md          
├── data
│   ├── external       
│   ├── interim        
│   ├── processed      
│   └── raw            
│
├── docs               
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         eeg_seizure_cls and configuration for tools like black
│
├── references         
│
├── reports            
│   └── figures        
│
├── requirements.txt   <- The requirements file generated with `pip freeze > requirements.txt`
│
├── setup.cfg          
│
└── eeg_seizure_cls   <- Source code for use in this project.
    │
    ├── __init__.py             
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

