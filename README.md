# Protein transfer learning improves identification of heat shock protein families (PLOS ONE 2021)
Official Pytorch implementation of **DeeperHSP** | [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251865)

## Abstract
Heat shock proteins (HSPs) play a pivotal role as molecular chaperones against unfavorable conditions. Although HSPs are of great importance, their computational identification remains a significant challenge. Previous studies have two major limitations. First, they relied heavily on amino acid composition features, which inevitably limited their prediction performance. Second, their prediction performance was overestimated because of the independent two-stage evaluations and train-test data redundancy. To overcome these limitations, we introduce two novel deep learning algorithms: (1) time-efficient DeepHSP and (2) high-performance DeeperHSP. We propose a convolutional neural network (CNN)-based DeepHSP that classifies both non-HSPs and six HSP families simultaneously. It outperforms state-of-the-art algorithms, despite taking 14-15 times less time for both training and inference. We further improve the performance of DeepHSP by taking advantage of protein transfer learning. While DeepHSP is trained on raw protein sequences, DeeperHSP is trained on top of pre-trained protein representations. Therefore, DeeperHSP remarkably outperforms state-of-the-art algorithms increasing F1 scores in both cross-validation and independent test experiments by 20% and 10%, respectively. We envision that the proposed algorithms can provide a proteome-wide prediction of HSPs and help in various downstream analyses for pathology and clinical research.

## How to Run
#### Example:
```
CUDA_VISIBLE_DEVICES=0 python embed_data.py --data-path data/ --model-config config/model/DeeperHSP.json
CUDA_VISIBLE_DEVICES=0 python train_model.py --data-config config/data/HSP_train.json --model-config config/model/DeeperHSP.json --run-config config/run/run.json --output-path results/DeeperHSP_final/
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --data-config config/data/HSP_test.json --model-config config/model/DeeperHSP.json --run-config config/run/run.json --checkpoint pretrained_models/DeeperHSP_final.pt --output-path results/DeeperHSP_final/
```
<br/>

## HSP Datasets
- <a href="https://www.dropbox.com/s/pdsuxboaehx8sgg/FASTA.tar.gz?dl=0">FASTA</a> : 
  FASTA files for generating OneHot/ESM embeddings (CV and Test datasets)
- <a href="https://www.dropbox.com/s/mstmdig8mv3es5t/OneHot_test.tar.gz?dl=0">OneHot</a> :
  DeepHSP OneHot embedding files (Test dataset) 
- <a href="https://www.dropbox.com/s/zs8jvm6el5r4k12/ESM_test.tar.gz?dl=0">ESM</a> :
  DeeperHSP ESM embedding files (Test dataset)
<br/>
  
Due to the large file sizes, we only provide OneHot & ESM embedding files for the Test dataset.<br/>
OneHot & ESM embeddings files for the CV dataset can be obtained from FASTA files using <code>embed_data.py</code> script. <br/>
<br/>

## Requirements
- Python 3.8
- PyTorch 1.5.1
- Bio Embeddings 0.1.5
- Numpy 1.20.1
- Scipy 1.6.0
- Scikit-Learn 0.24.1
- Thop 0.0.31
<br/><br/><br/>
