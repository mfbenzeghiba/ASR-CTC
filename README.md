# ASR-CTC
This repos uses PyTorch to fine-tune state-of-the-art pretrained foundational ASR models, including Wav2Vec2.0, Hubert and WavLM.
The fine tuning is done by adding a linear CTC layer at the output of the pretrained models, and finetuning the original encoder
with the added CTC layer. The feature extraction backbone is kept as it is (without finetuning it).
Initial experiments are performed on TIMIT databse, with the task of recognizing characters.

## Evaluation

The table belows reports the performance in terms of CER% (Character Error Rate) and WER (Word Error Rate) in End-to-End scenario (without language model).

## Dev data

|       | CER (%)  |  WER |
|:------|--------:|--------:|
| Wav2Vec2 | **6.8** | **25.9** |
| Hubert  | 7.0 | 27.7 |
| WavLM | 7.0 | 27.0 |


## Test data

|       | CER (%)  |  WER |
|:------|--------:|--------:|
| Wav2Vec2 | 7.2 | **26.9** |
| Hubert  | 7.9 | 30.0 |
| WavLM | **7.1** | 27.4 |
