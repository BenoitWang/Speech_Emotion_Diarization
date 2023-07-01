<p align="center" width="100%">
<img src="images/ser_sed.png" alt="ser_sed" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>


# Speech Emotion Diarization (SED)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://https://github.com/BenoitWang/Speech_Emotion_Diarization/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://https://github.com/BenoitWang/Speech_Emotion_Diarization/blob/main/DATA_LICENSE)

Speech Emotion Diarization ([arXiv link](https://arxiv.org/pdf/2306.12991.pdf)) aims to predict the correct emotions and their temporal boundaries within an utterance. 


## Dependencies

The implementation is based on the popular speech tookit [SpeechBrain](https://github.com/speechbrain/speechbrain). 

Another implementation will be made available soon on SpeechBrain.


To install the dependencies, do  `pip install -r requirements.txt`



## Datasets

### Test Set
The test is based on Zaion Emotion Dataset (ZED), which can be downloaded [here](https://zaion.ai/en/resources/zaion-lab-blog/zaion-emotion-dataset/).

### Training Set Preparation
1. RAVDESS: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1

   Unzip and rename the folder as "RAVDESS".

2. ESD: https://github.com/HLTSingapore/Emotional-Speech-Data

   Unzip and rename the folder as "ESD".

3. IEMOCAP: https://sail.usc.edu/iemocap/iemocap_release.htm

   Unzip.

4. JL-CORPUS: https://www.kaggle.com/datasets/tli725/jl-corpus?resource=download

   Unzip, keep only `archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)` and rename the folder to "JL_corpus".

5. EmoV-DB: https://openslr.org/115/

   Download `[bea_Amused.tar.gz, bea_Angry.tar.gz, bea_Neutral.tar.gz, jenie_Amused.tar.gz, jenie_Angry.tar.gz, jenie_Neutral.tar.gz, josh_Amused.tar.gz, josh_Neutral.tar.gz, sam_Amused.tar.gz, sam_Angry.tar.gz, sam_Neutral.tar.gz]`, unzip and move all the folders into another folder named "EmoV-DB".



## Metric

A proposed Emotion Diarization Error Rate is used to evaluate the baselines.
![EDER](images/EDER.png)
The four components are:

1. False Alarm (FA): Length of non-emotional segments that are predicted as emotional.
2. Missed Emotion (ME): Length of emotional segments that are predicted as non-emotional.
3. Emotion Confusion (CF): Length of emotional segments that are assigned to another(other) incorrect emotion(s).
4. Emotion Overlap (OL): Length of non-overlapped emotional segments that are predicted to contain other overlapped emotions apart from the correct one



## Run the code

Model configs and experiment settings can be modified in `hparams/train.yaml`.



To run the code, do `python train.py hparams/train.yaml --zed_folder /path/to/ZED --emovdb_folder /path/to/EmoV-DB --esd_folder /path/to/ESD --iemocap_folder /path/to/IEMOCAP --jlcorpus_folder /path/to/JL_corpus --ravdess_folder /path/to/RAVDESS`.



The data preparation may take a while.



A `results` repository will be generated that contains checkpoints, logs, etc. The frame-wise classification result for each utterance can be found in `eder.txt`.



## Inference

The pretrained models and a easy-inference interface can be found on [HuggingFace](to be added).



## **About Speech Emotion Diarization/Zaion Emotion Dataset**

```bibtex
@article{wang2023speech,
  title={Speech Emotion Diarization: Which Emotion Appears When?},
  author={Wang, Yingzhi and Ravanelli, Mirco and Nfissi, Alaa and Yacoubi, Alya},
  journal={arXiv preprint arXiv:2306.12991},
  year={2023}
}
```
