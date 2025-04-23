"""data provider for TIMIT data base."""

import random
from typing import List, Dict, Tuple

import regex
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio


class TimitTokenizer():
    """Create timit vocabulary."""

    def __init__(self, data_config: dict) -> None:
        """Create timit tokenizer

        Args:
            data_config (dict): data configuration
        """

        self.token_to_index = {}
        self.index_to_token = {}
        self.blank_token = '<blank>'
        self.blank_idx = 0
        self.modeling_unit = data_config.modeling_unit
        self.map_key = None
        
        if self.modeling_unit == 'phoneme':
            if data_config.phone_mapping_file is not None:
                self.phone_mapping = self.read_phone_mapping(data_config.phone_mapping_file)
                self.map_key = data_config.mapping_key
                self.build_phoneme_vocab(self.map_key)
            else:
                print('The phone mapping file is missing....')
        else:
            if data_config.char_vocab is not None:
                self.build_char_vocab(data_config.char_vocab)
            else:
                print('The character vocab file is missing....')


    def build_phoneme_vocab(self, map_key: str = None) -> None:
        """Build the vocab.

        Args:
            map_key (str): The phone mapping key.
        """

        if self.modeling_unit == 'phoneme' and map_key is not None:
            self.map_key = map_key

        pm = self.phone_mapping[self.map_key]
        self.index_to_token = {}
        self.token_to_index = {}

        for j, token in enumerate(sorted(set(pm.values()))):
            if j >= self.blank_idx:
                j += 1
            self.index_to_token[j] = token
            self.token_to_index[token] = j
        self.index_to_token[self.blank_idx] = self.blank_token
        self.index_to_token = dict(sorted(self.index_to_token.items()))

        self.token_to_index = {v:k for k,v in self.index_to_token.items()}


    def read_phone_mapping(self, phone_mapping_file: str) -> Dict[str, Dict[str, str]]:
        """Read the phone mapping fot timit.

        Args:
            file (str): The phne mapping file.

        Returns:
            dict: The phone mapping.
        """

        pm = dict()
        with open(phone_mapping_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [l.strip().split() for l in lines]
            pm['61to61'] = {line[0]: line[0] for line in lines if len(line) == 3}
            pm['61to48'] = {line[0]: line[1] for line in lines if len(line) == 3}
            pm['48to39'] = {line[1]: line[2] for line in lines if len(line) == 3}
            pm['61to39'] = {line[0]: line[2] for line in lines if len(line) == 3}
        return pm

    def build_char_vocab(self, char_vocab: str) -> None:
        """Read the char vocabulary.

        Args:
            char_vocab (str): The char vocbulary file
        """

        with open(char_vocab, 'r', encoding='utf-8') as f:
            for i, line in enumerate( f.readlines() ):
                token = line.strip()
                if token == '<space>':
                    token = ' '
                if i >= self.blank_idx:
                    i += 1
                self.index_to_token[i] = token
                self.token_to_index[token] = i
        self.index_to_token[self.blank_idx] = self.blank_token
        self.index_to_token = dict(sorted(self.index_to_token.items()))
        self.token_to_index = {v:k for k,v in self.index_to_token.items()}

    def __len__(self) -> int:
        """Return the length of the vocab

        Returns:
            int: Number of tokens
        """

        return len(self.index_to_token)

    @property
    def vocab_itot(self) -> Dict[int, str]:
        """Return the index to token dictionnary."""

        return self.index_to_token

    @property
    def vocab_ttoi(self) -> Dict[str, int]:
        """Return the token to index dictionnary."""

        return self.token_to_index


    def encode(self, trans: List[str]) -> torch.Tensor:
        """Convert a text to a sequence of token index

        Args:
            trans (str): The transcription to be encoded.

        Returns:
            torch.tensor: _description_
        """

        if self.modeling_unit == 'phoneme':
            trans = [self.phone_mapping[self.map_key][token] for token in trans]
        vector = [self.token_to_index[token] for token in trans]
        return torch.LongTensor(vector)


    def decode(self, labs: List[int]) -> List[str]:
        """Convert a sequence of integer labels to a text.

        Args:
            labels (torch.tensoer): sequence to be decoded.

        Returns:
            List: a text, sequence of tokens
        """

        tokens = [self.index_to_token[l] for _, l in enumerate(labs)]
        return tokens


class TimitDataset(Dataset):
    """Define data loader."""

    def __init__(
            self,
            data_file: str,
            tokenizer,
            replicate: int = 1
        ):

        self._data_samples = pd.read_csv(data_file)
        if replicate > 1:
            data = [self._data_samples]*replicate
            self._data_samples = pd.concat(data, axis=0, ignore_index=True)
        self.tokenizer = tokenizer


    def __len__(self) -> int:
        """Return the length of the data

        Returns:
            int: Number of sentences
        """
        return len(self._data_samples)

    def __getitem__(self, index) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Return the sample (wav_file and targets) with index index

        Args:
            index (int): The index of the sample

        Returns:
            tuple: The wavfile and the corresponding phone transcription
        """
        x, y, z = self._data_samples.iloc[index]
        signal, _ = torchaudio.load(x)
        signal = signal.squeeze(0)
        if self.tokenizer.modeling_unit == 'phoneme':
            trans = y.split(' ')
        else:
            trans = regex.sub('[?,.\\\\!;:"-]', '', z).lower()
            trans = list(trans)    
        trans = self.tokenizer.encode(trans)
        trans = trans.squeeze(0)
        return x, signal, trans


    def fn_collate(self, batch):
        """Generate a mini-batch of data. For DataLoader's 'collate_fn'.

        Args:
            batch (list(tuple)): A mini-batch of (wavforms, label sequences) pairs.

        Returns:
            xs (list, [batch_size, (padded) seq_length, dim_features]): A mini-batch of wavforms.
            ys (torch.LongTensor, [batch_size, (padded) n_tokens]): A mini-batch of label sequences.
            ylens (torch.LongTensor, [batch_size]): Sequence lengths before padding.
        """

        audio_files = []
        waveforms = []
        targets = []

        for _, data in enumerate(batch):
            audio_files.append(data[0])
            waveforms.append(data[1])
            targets.append(data[2])

        waveform_sizes = torch.IntTensor([wf.shape[0] for wf in waveforms])
        waveforms = pad_sequence(waveforms, batch_first=True)
        target_sizes = torch.IntTensor([tar.shape[0] for tar in targets ])
        targets = pad_sequence(targets, batch_first=True)   # [batch_size, (padded) n_tokens]
        return audio_files, waveforms, waveform_sizes, targets, target_sizes


if __name__ == "__main__":

    random.seed(10)

    TRAIN_FILE = 'path/to/processed_train.csv'
    DATA_DIR = 'path/to/data/'
    PHONE_MAPPING = 'path/to/data/phones.60-48-39.map.txt'
    VOCAB_CHAR = r'C:/Users/Mohammed/my_work/data/timit/vocab_char.txt'
    MODELING_UNIT = 'phoneme'
    MAPPING_KEY = '61to39'
    BATCH_SIZE = 4
    WORKERS = 0
    SAMPLE_RATE = 16000

    phoneme_cfg = OmegaConf.create(
        {
            'mapping_key': MAPPING_KEY,
            'phone_mapping_file':  PHONE_MAPPING,
            'modeling_unit': MODELING_UNIT,
            'blank_token': '<blank>',
            'blank_idx': 0
        }
    )

    char_cfg = OmegaConf.create(
        {
            'char_vocab': VOCAB_CHAR,
            'modeling_unit': MODELING_UNIT,
            'blank_token': '<blank>',
            'blank_idx': 0
        }
    )

    features_config = OmegaConf.create(
        {
            'feature_type': 'mfcc',
            'use_energy': False,
            'feature_dim': 39,
            'add_noise': False
        }
    )

    if MODELING_UNIT == 'phoneme':
        ttokenizer = TimitTokenizer(data_config=phoneme_cfg)
    else:
        ttokenizer = TimitTokenizer(data_config=char_cfg)
    num_classes = len(ttokenizer)
    print(num_classes)
    print(ttokenizer.token_to_index)
    dataset = TimitDataset(TRAIN_FILE, ttokenizer)

    data_loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        collate_fn=dataset.fn_collate,
                        shuffle=False,
                        num_workers=WORKERS,
                        pin_memory=True)

    print(f'Length of dataset: {len(data_loader)}')
    print(f'Vocabulary size: {len(dataset.tokenizer)}')

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(BATCH_SIZE, 1)
    for _, b in enumerate(data_loader):
        audio_file, wforms, wforms_lengths, labels, labels_lengths = b
        for k, wf in enumerate(wforms):
            print(audio_file[k])
            print(ttokenizer.decode(labels.tolist()[k]))
            waveform = wforms[k]
            num_frames = waveform.shape[0]
            time_axis = torch.arange(0, num_frames) / SAMPLE_RATE
            axes[k].plot(time_axis, waveform, linewidth=1)
            figure.suptitle('Wave formes of the batch audio files')
        plt.show()
        exit()
