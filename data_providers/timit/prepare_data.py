"""process timit data.
https://github.com/biyoml/PyTorch-End-to-End-ASR-on-TIMIT/blob/master/
"""

import os
from typing import List
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

SPEAKERS_TEST = [
    'MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0',
    'MJMP0', 'MLNT0', 'FPKT0', 'MLLL0', 'MTLS0', 'FJLM0',
    'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0',
    'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0'
    ]


def read_phonems(file_list: List) -> List:
    """Read Timit phone files

    Args:
        file_list (List): List of phonetic files
    """

    phonetic_trans = []
    for wf in file_list:
        pf = wf.replace('.WAV', '.PHN')
        with open(pf, 'r', encoding='utf8') as f:
            phonemes = f.readlines()
            ptrans = [l.strip().split()[-1] for l in phonemes if l.strip().split()[-1] != 'q']
            phonetic_trans.append(' '.join(ptrans))
    return phonetic_trans

def read_dev_speakers(dev_speakers_file: str) -> List:
    """Read the dev speakers list

    Args:
        dev_speakers_file (str): Path to the file containing dev speakers list

    Returns:
        List: list of dev speakers
    """

    with open(dev_speakers_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.strip().split('/')[-1].upper() for l in lines]
    lines = [l.split('.')[0] for l in lines]
    return lines

def read_text(file_list: List) -> List:
    """Read text file

    Args:
        file_list (list): _description_
    """

    text_trans = []
    for wf in file_list:
        tf = wf.replace('.WAV', '.TXT')
        with open(tf, 'r', encoding='utf-8') as f:
            text = f.readline().strip()
            text = ' '.join(text.split()[2:])
            if "\"" in text:
                print(text)
                print(tf)
            text_trans.append(text)
    return text_trans

def process_dataset(data_dir: str, data_file: str, split: str = 'train',
                    dev_speakers_file : str = None):
    """Process the Timit dataset.

    Args:
        data_dir (str): The data folder
        data_file (str): The file list
        split (str, optional): The split (train test)
        dev_speakers_file (str): file containing the list of dev speakers.        
    """

    logging.info('Preparing data for the %s split...', split)

    if dev_speakers_file is not None:
        dev_speakers = read_dev_speakers(dev_speakers_file)
        logging.info('Number of dev speakers: %d', len(dev_speakers))

    data_fields = pd.read_csv(data_file)

    # select the data
    if split == 'train':
        audio_df = data_fields[data_fields['filename'].str.endswith('.WAV', na=False)]
    else:
        if split == 'test':
            audio_df = data_fields[data_fields['speaker_id'].isin(SPEAKERS_TEST)
                                   & data_fields['filename'].str.endswith('.WAV')]
        elif split == 'valid':
            audio_df = data_fields[~data_fields['speaker_id'].isin(SPEAKERS_TEST)
                                   & data_fields['filename'].str.endswith('.WAV')]
            indexes = []
            for i in range(len((audio_df))):
                row = audio_df.iloc[i]
                speaker_id = row['speaker_id']
                filename = row['filename'].split('.')[0]
                dev_speaker = f'{speaker_id}_{filename}'.upper()
                if dev_speaker not in dev_speakers:
                    indexes.append(i)

            audio_df.drop(audio_df.index[indexes], inplace=True)

    # Remove all 'SA' records.
    logging.info('Remove all SA records...')
    audio_df = audio_df[~audio_df['filename'].str.startswith('SA')]
    wav_list = [os.path.join(data_dir, audio_df.iloc[index, 6]) for index in range(len(audio_df))]
    logging.info('Read phonemes filles...')
    phones_transcription = read_phonems(wav_list)
    logging.info('Read text transcriptions...')
    text_transcription = read_text(wav_list)

    data = {
        'audio':wav_list,
        'phonemes_transcription': phones_transcription,
        'text_transcription': text_transcription
    }
    # save the data
    fdata = pd.DataFrame(data)
    fname = os.path.join(data_dir, f'processed_{split}.csv')
    fdata.to_csv(fname, index=False)
    logging.info('Data preparation is finished...')
    logging.info('The file %s is created...', fname)
    return fdata


if __name__ == "__main__":

    TRAIN_FILE = 'path/to/timit/train_data.csv'
    TEST_FILE = 'path/to/timit/test_data.csv'
    DATA_DIR = r'path/to/timit/data'
    DEV_SPEAKERS_LIST = 'path/to/data/dev_speaker_list.txt'

    processed_train = process_dataset(data_dir=DATA_DIR,
                                     data_file=TRAIN_FILE,
                                     split='train')

    logging.info('Number of train sentences %d', len(processed_train))
    processed_valid = process_dataset(data_dir=DATA_DIR,
                                      data_file=TEST_FILE,
                                      split='valid',
                                      dev_speakers_file=DEV_SPEAKERS_LIST)
    logging.info('Number of valid sentences %d ', len(processed_valid))

    processed_test = process_dataset(data_dir=DATA_DIR,
                                     data_file=TEST_FILE,
                                     split='test')
    logging.info('Number of test sentences %d', len(processed_test))
