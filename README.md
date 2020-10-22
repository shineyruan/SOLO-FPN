# SOLO-FPN
SOLO instance segmentation with Feature Pyramid networks. For original authors' code see [here.](https://github.com/WXinlong/SOLO)

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

To train the network, run
```bash
python src/main_train.py
```

To test the network, run
```bash
python src/main_infer.py
```



## References
> [**SOLO: Segmenting Objects by Locations**](https://arxiv.org/abs/1912.04488),            
> Xinlong Wang, Tao Kong, Chunhua Shen, Yuning Jiang, Lei Li    
> In: Proc. European Conference on Computer Vision (ECCV), 2020  
> *arXiv preprint ([arXiv 1912.04488](https://arxiv.org/abs/1912.04488))*   


> [**SOLOv2: Dynamic, Faster and Stronger**](https://arxiv.org/abs/2003.10152),            
> Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, Chunhua Shen        
> *arXiv preprint ([arXiv 2003.10152](https://arxiv.org/abs/2003.10152))*  
