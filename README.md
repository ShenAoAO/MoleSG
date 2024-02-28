# MoleSG

## 🤖 Introduction
In this study, we propose an effective multi-modality self-supervised learning framework for molecular SMILES and Graph. Specifically, SMILES data and graph data are first tokenized so that they can be processed by a unified Transformer-based backbone network, which is trained by a masked reconstruction strategy. In addition, we introduce a specialized non-overlapping masking strategy to encourage fine-grained interaction between these two modalities.

The code was built based on [CoMPT](https://github.com/jcchan23/CoMPT) and [Chemberta](https://github.com/seyonechithrananda/bert-loves-chemistry). Thanks a lot for their code sharing!

## 🔬 Dependencies

+ cuda >= 9.0
+ cudnn >= 7.0
+ RDKit == 2020.03.4
+ torch >= 1.4.0 (please upgrade your torch version in order to reduce the training time)
+ numpy == 1.19.1
+ scikit-learn == 1.3.0
+ tqdm == 4.52.0
+ transformers == 4.31.0
+ torch-geometric == 1.7.2

Tips: Using code `conda install -c conda-forge rdkit` can help you install package RDKit quickly.

## 📚 Dataset

|    Dataset    | Tasks | Type | Molecule | Metric | 
|:-------------:|:-----:| :---: |:--------:| :---: |
|     bbbp      |   1   | Graph Classification |  2,035   | ROC-AUC |
|     tox21     |  12   | Graph Classification |  7,821   | ROC-AUC |
|    ToxCast    |  617  | Graph Classification |  8,575   | ROC-AUC | 
|     sider     |  27   | Graph Classification |  1,379   | ROC-AUC |
|    clintox    |   2   | Graph Classification |  1,468   | ROC-AUC |
|     bace      |   1   | Graph Classification |  1,513   | ROC-AUC |
|     esol      |   1   | Graph Regression |  1,128   | RMSE |
|   freesolv    |   1   | Graph Regression |   642    | RMSE |
| lipophilicity |   1   | Graph Regression |  4,198   | RMSE |
|      QM7      |   1   | Graph Regression |  6,830   | MAE |
|      QM8      |  12   | Graph Regression |  21,786  | MAE |
|      QM9      |   3   | Graph Regression | 133,885  | MAE |

## 📚 Data preprocess

For the original pre-training dataset, you can download the source dataset from [Molecule-Net](http://moleculenet.ai/datasets-1).

For the original downstream dataset, you can download the source dataset from [ZINC15](https://github.com/HICAI-ZJU/KANO/blob/main/data/zinc15_250K.csv).

For your convenience, we provide our processed data and our process code for each dataset in https://drive.google.com/file/d/16MHQk8AkmyqqCI1r0vb4S9o5DT8t7dd_/view?usp=sharing.

## 🚀 Pre-training
If you want to retrain our pre-train model, you can run:
```sh
>> python train_total.py \
    --experiment_name pretrain_test \
    --epochs 30
```

We provide our pre-trained model in https://drive.google.com/file/d/1BsJyZeBfvl5QMp3gj4EBcwUu3e1Waazm/view?usp=sharing
## 🚀 Fine-tuning

Note that if you change the downstream benchmark, don't forget to change the corresponding `dataset` and `split`! For example:
```sh
>> python train_graph.py \
    --experiment_name test \
    --gpu 0 \
    --fold 1 \
    --dataset bbbp \
    --split scaffold \
    --gpu 1 \
    --ckpt_path 'your_pretrained_model_path'
```
where `<seed>` is the seed number, `<gpu>` is the gpu index number, `<split>` is the split method (except for qm9 is random, all are scaffold), `<dataset>` is the element name('bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'bace', 'muv', 'hiv','esol', 'freesolv', 'lipophilicity','qm7','qm8', 'qm9').

All hyperparameters can be tuned in the `utils.py`

## 💡 Todo

- [x] Provide our ablation study codes.

