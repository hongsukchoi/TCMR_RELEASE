# Data pre-procseeing

- The 'pre-processing' indicates to compute and to save image features before training and testing as written in our [paper](https://arxiv.org/abs/2011.08627).
- The pre-processed data we provide in this [link](https://drive.google.com/drive/folders/1h0FxBGLqsxNvUL0J43WkTxp7WgYIBLy-?usp=sharing) are processed following the VIBE repository's [instruction](https://github.com/mkocabas/VIBE/blob/master/doc/train.md).
- Download the data from sources following the [instruction](https://github.com/mkocabas/VIBE/blob/master/doc/train.md), and run our python scripts as below
```bash
python lib/data_utils/{dataset_name}_utils.py --dir ./data/{dataset_name}
```
- You may need to change details (ex. scale), so check comments in `{dataset_name}_utils.py` files.
- To get occluusion-augmented train data following our [paper](https://arxiv.org/abs/2011.08627) (inspired by this [paper](https://arxiv.org/abs/1808.09316)), 
    - First, download VOC2012 data and locate under `${ROOT}/data/`.
    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xf VOCtrainval_11-May-2012.tar
    ```
    - Second, uncomment `load_occluder` function and pass the `occluders` variable to `extract_features` function in `{dataset_name}_utils.py`.
    - Last, set `occ` argument in `get_single_image_crop` function in `_feature_extractor.py`.
- We did not pre-processed InstarVariety dataset. We used the data pre-processed by VIBE. 

If you have a problem with 'download limit' when trying to download datasets from google drive links, please try this trick.
>* Go the shared folder, which contains files you want to copy to your drive  
>* Select all the files you want to copy  
>* In the upper right corner click on three vertical dots and select “make a copy”  
>* Then, the file is copied to your personal google drive account. You can download it from your personal account.  