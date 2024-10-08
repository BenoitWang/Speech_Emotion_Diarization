<p align="center" width="100%">
<img src="images/ser_sed.png" alt="ser_sed" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>


# Speech Emotion Diarization (SED)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://https://github.com/BenoitWang/Speech_Emotion_Diarization/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://https://github.com/BenoitWang/Speech_Emotion_Diarization/blob/main/DATA_LICENSE)

[Speech Emotion Diarization](https://arxiv.org/pdf/2306.12991.pdf) is a technique that focuses on predicting emotions and their corresponding time boundaries within a speech recording. 

The model has been trained using audio samples that include one non-neutral emotional event, which belong to one of the four following transitional sequences:
```
neutral-emotional
neutral-emotional-neutral
emotional-neutral
emotional
```


The model's output takes the form of a dictionary comprising emotion components (*neutral*, *happy*, *angry*, and *sad*) along with their respective start and end boundaries, as exemplified below:

```python
{
   'example.wav': [
      {'start': 0.0, 'end': 1.94, 'emotion': 'n'},  # 'n' denotes neutral
      {'start': 1.94, 'end': 4.48, 'emotion': 'h'}   # 'h' denotes happy
   ]
}
```


## Dependencies

The implementation is based on the popular speech tookit [SpeechBrain](https://github.com/speechbrain/speechbrain).

Another implementation of this project can be found [here](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ZaionEmotionDataset) as a SpeechBrain recipe.

To install the dependencies, do  `pip install -r requirements.txt`



## Datasets

### Test Set
The test is based on **Zaion Emotion Dataset (ZED)**, which can be downloaded via this [link](https://www.dropbox.com/scl/fi/2s3ro8tmgt1lir77z3hj5/ZED.zip?rlkey=qkizx7t3ozo02xs7k1tlexb1e&st=9l466c1c&dl=0).

### Training Set Preparation
1. RAVDESS: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1

   Unzip and rename the folder as "RAVDESS".

2. ESD: Download the ESD dataset via this [link](https://drive.google.com/file/d/1_0qvBnrrDPWqFiGi4JThm0X5fYTu-Fm4/view?usp=sharing). It should be noted that the `prepare_ESD.py` script works only with this old version of the dataset.

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

Even though frame-wise classification accuracy can also reflect the system's capacity, it is not always convincing because it depends on the frame length (resolution). A higher accuracy of frame-wise classification does not equal that the model can better diarize. Hence, EDER is a more common metric for the task.

## Run the code

Model configs and experiment settings can be modified in `hparams/train.yaml`.


To run the code, do `python train.py hparams/train.yaml --zed_folder /path/to/ZED --emovdb_folder /path/to/EmoV-DB --esd_folder /path/to/ESD --iemocap_folder /path/to/IEMOCAP --jlcorpus_folder /path/to/JL_corpus --ravdess_folder /path/to/RAVDESS`.



The data preparation may take a while.



A `results` repository will be generated that contains checkpoints, logs, etc. The frame-wise classification result for each utterance can be found in `eder.txt`.



## Results

The EDER (Emotion Diarization Error Rate) reported here was averaged on 5 different seeds, results of other models (wav2vec2.0, HuBERT) can be found in the paper. You can find our training results (model, logs, etc) [here](https://www.dropbox.com/sh/woudm1v31a7vyp5/AADAMxpQOXaxf8E_1hX202GJa?dl=0).


| model | EDER |
|:-------------:|:---------------------------:|
| WavLM-large | 30.2 ± 1.60 |


It takes about 40 mins/epoch with 1xRTX8000(40G), reduce the batch size if OOM.



## Inference

The pretrained models and a easy-inference interface can be found on [HuggingFace](https://huggingface.co/speechbrain/emotion-diarization-wavlm-large).



## **About Speech Emotion Diarization/Zaion Emotion Dataset**

```bibtex
@inproceedings{wang2023speech,
  title={Speech emotion diarization: Which emotion appears when?},
  author={Wang, Yingzhi and Ravanelli, Mirco and Yacoubi, Alya},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={1--7},
  year={2023},
  organization={IEEE}
}
```
