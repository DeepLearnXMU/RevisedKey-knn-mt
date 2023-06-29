# RevisedKey-knn-mt

Code for our ACL 2023 paper "Bridging the Domain Gaps in Context Representations for k-Nearest Neighbor Neural Machine Translation". 

The source code is developed upon kNN-MT. You can see detail in https://github.com/urvashik/knnmt, many thanks to the authors for making their code avaliable.

## Requirements and Installation

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0

You can install this project by
```
pip install --editable ./
```

## Instructions

We use an example to show how to use our codes.

### Pre-trained Model and Data

The pre-trained translation model can be downloaded from [this site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md).
We use the De->En Single Model for all experiments.

For convenience, the pre-processed data can be downloaded from [this site](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view?usp=sharing).

### Build Datastore

This script will create datastore (includes key.npy and val.npy) for the data.

```
cd revisedkey-scripts
# use pre-trained model to build datastore for koran domain (the following `base` means the pre-trained model)
bash build_datastore.sh base koran
```

### Evaluating kNN-MT with Base Model

To evaluate the kNN-MT on the test set:

```
cd revisedkey-scripts
bash knnmt_inference.sh base koran
```

### Train Our Model

To train the revised:

```
cd revisedkey-scripts
# step 1. prepare heldout data from base model
bash save_retrieve_result.sh base koran
bash save_valid_kv.sh base koran
bash save_valid_retrieve_result.sh base koran

# step 2. fine-tune base model to prepare heldout data
bash finetune.sh koran

# step 3. prepare heldout data from fine-tuned model
bash build_datastore.sh finetune koran
bash save_retrieve_result.sh finetune koran
bash save_valid_kv.sh finetune koran
bash save_valid_retrieve_result.sh finetune koran

# step 4. train reviser
bash train_reviser.sh koran
```

Or you can obtain the revised datastores following steps:
1. Download checkpoints of fine-tuned models from [this site](https://drive.google.com/file/d/1vaftBeajBj3VWIYAdqLqWqR5GCiYk1VN/view?usp=sharing)
2. Download trained revisers and faiss index from [this site](https://drive.google.com/file/d/1XAPJTANXXNjNZGjy1DVLjc5_Y8R_GF2Z/view?usp=sharing)
3. Refer [datastore-revise.ipynb](https://github.com/DeepLearnXMU/RevisedKey-knn-mt/blob/main/datastore-revise.ipynb) to revise the original datastore


### Evaluating with Our Method

```
cd revisedkey-scripts
bash batch_knnmt_inference.sh koran ../save/news_to_koran
```
