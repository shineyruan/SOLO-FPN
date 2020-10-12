# SOLO-FPN
SOLO instance segmentation network

*University of Pennsylvania - CIS 680 FA 2020 - Advanced Topics in Machine Perception*

## Usage

In order to run the whole program, you have to first install corresponding packages as specified in `environment.yml`. If you are using `conda`, you can simply run 
```bash
conda env create -f environment.yml
```

To test the dataset builder and loader, run 
```bash
python src/dataset.py
```

To test the SOLO-head, run
```bash
python src/solo_head.py
```
