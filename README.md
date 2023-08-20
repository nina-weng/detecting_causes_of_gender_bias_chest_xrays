# Detecting the Causes of Gender Bias in Chest X-ray Diagnosis

Code repository for [2023 MICCAI workshop: FAIMI paper](https://arxiv.org/abs/2308.05129)

'''
tmp for integrating:
functions: 
1. being able to access both datasets
2. only the resampling dataloader
3. run command: python train.py -ds dataset -d disease_label -fp female_percent_in_training -npp number_per_patient -rs random_states
    and some other defaulted ones (could be set during training): -lr -epochs -model -model_scale -pretrained -aug -is_multilabel -image_size -crop
    -prevalence_setting -save_model
'''
## Preprocessing

{pre-processing the image: resize to 224 x 224
pre-processing the mata data}

## Training
`python3 ./disease_prediction.py -s NIH -d Pneumothorax -f 0 50 100 -n 1 -r 0-10`
Details about the hyper-parameters could be found in the same py file.

## Plotting the results

need to create a new py for it

## Detecting the Causes
### 1. Representation Bias (Imbalance Dataset)

### 2. Feature Bias (Breast shadows)

{the experiment about cropping the image}


### 3. Label Errors 

change the sampling strategies: run which py
cross sampling strategies experiments: run which py

