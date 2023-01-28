# Emotion Diarization
This project aims to accomplish a emotion diarization task, the solution is tested on ZED (Zaion Emotion Database).

## Dependencies
The implementation is based on SpeechBrain
To run the recipe, do `python train_with_wav2vec.py hparams/train_with_wav2vec.yaml`
The preparation may take several hours.

## Datasets
### Test Set
ZED (Zaion Emotion Database), download link to be added

### Training Set
1. RAVDESS: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1
    Unzip and rename the folder as "RAVDESS"
2. ESD: https://github.com/HLTSingapore/Emotional-Speech-Data
    Unzip and rename the folder as "ESD"
3. IEMOCAP: https://sail.usc.edu/iemocap/iemocap_release.htm
    Unzip
4. JL-CORPUS: https://www.kaggle.com/datasets/tli725/jl-corpus?resource=download
    Unzip, keep only `archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)` and rename the folder to "JL_corpus"
5. EmoV-DB: https://openslr.org/115/
    Download `[bea_Amused.tar.gz, bea_Angry.tar.gz, bea_Neutral.tar.gz, jenie_Amused.tar.gz, jenie_Angry.tar.gz, jenie_Neutral.tar.gz, josh_Amused.tar.gz, josh_Neutral.tar.gz, sam_Amused.tar.gz, sam_Angry.tar.gz, sam_Neutral.tar.gz]`, unzip and move all the folders into another folder named "EmoV-DB"
