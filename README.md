# Protein Transfer Learning Improves Identification of Heat Shock Protein Families

## Abstract
<p style="text-align:justify">
<strong>Motivation:</strong> Heat shock proteins (HSPs) play a pivotal role as molecular chaperones against unfavorable conditions. Although HSPs hold great importance, their computational identification remains a significant challenge. Previous studies have two major limitations. First, they relied heavily on amino acid composition features, which inevitably limited their prediction performance. Furthermore, their prediction performance was overestimated because of the independent two-stage evaluations and train-test data redundancy.
<br/>
<strong>Results:</strong> To overcome the previous limitations, we introduce two novel deep learning algorithms: (1) time-efficient DeepHSP and (2) high-performance DeeperHSP. First, we propose a convolutional neural network-based DeepHSP that classifies both non-HSPs and the six HSP families simultaneously. It outperforms the state-of-the-art algorithms and takes 14-15 times less time for both training and inference. Then, we further improve the performance of DeepHSP by taking advantage of protein transfer learning. In contrast to DeepHSP trained on raw protein sequences, we train DeeperHSP on top of the pre-trained protein representations. DeeperHSP remarkably outperforms the state-of-the-art algorithms in both cross-validation and independent test experiments, increasing the F1 score by 20% and 10%, respectively. We envision that the proposed algorithms can provide proteome-wide prediction HSPs and help various downstream analyses for pathology and clinical research.
<br/><br/>
</p>
<br/>

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
