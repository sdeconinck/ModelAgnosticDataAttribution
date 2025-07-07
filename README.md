# ModelAgnosticDataAttribution

Source code for the 'Mitigating Bias Using Model-Agnostic Data Attribution' paper, accepted to 2024 IEEE CVPR WORKSHOP ON FAIR, DATA-EFFICIENT, AND TRUSTED COMPUTER VISION

https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/html/De_Coninck_Mitigating_Bias_Using_Model-Agnostic_Data_Attribution_CVPRW_2024_paper.html 
## Installation

Install all necessary packages through `pip install -r requirements.txt`

## Data Setup

CelebA dataset can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Place under `data/celeba`.

## Training a region classifier

To train a model for region classification use the following command:

```
    python3 train.py --attribute Blond_Hair --n_regions 2401 --save_path models/model_blond_hair.pt
```

## Calculating attributions using the region classifier

```
python3 attribute_data.py --model_path "models/model_blond_hair.pt" --partition val --save_path attributions/blond_hair_val.pt
```

## Training a robust classifier using the region attributions

Use robustness_experiment.py to examine the difference between models trained with or without region attribution-based noise.
