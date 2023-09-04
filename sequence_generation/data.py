import os
import requests
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Dataset for senteces from the works of Shakespeare
"""

# Works of Shakespeare in txt format from MIT
DATA_SOURCE = 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt'
FILE_LOCATION = './sequence_generation/shakespeare.txt'

def download_and_parse_shakespeare(url, savefile):
    print(f"Downloading Shakespeares works from {url}")
    r = requests.get(url)
    with open(savefile, 'w') as f:
        is_header = True
        for line in r.content.split(b"\n"):
            text_line = line.decode("utf-8").strip()
            if is_header:
                # Skip first lines of file information
                if text_line == 'by William Shakespeare':
                    is_header = False
            else:
                # Only include non-empty, non-numeric and non-capital sentences
                if text_line != '' and not text_line.isnumeric() and not text_line.upper() == text_line and not text_line.startswith("End of this Etext"):
                    # Remove unnecessary characters
                    for c in [",",";","[","]"]:
                        text_line = text_line.replace(c,"")
                    # Remove words in all capitals
                    parsed_text = " ".join([w for w in text_line.split(" ") if w.upper() != w] + ["\n"])
                    f.write(parsed_text)


class RandomShakespeareSentence(Dataset):
    """
    Dataset that outputs encoded sequences from the works of Shakespeare
    """
    def __init__(self, max_length=None):
        if not os.path.exists(FILE_LOCATION):
            download_and_parse_shakespeare(DATA_SOURCE, FILE_LOCATION)

        self.sentence_lines = []
        with open(FILE_LOCATION) as f:
            for line in f:
                for c in ["\n","<",">","(",")",'"',"'"]:
                    line = line.replace(c,"")
                self.sentence_lines.append(line.lower().strip())

        self.start_token = 0
        self.max_length = max_length or max(map(len, self.sentence_lines))
        # self.characters = list(set([c for sentence in self.sentence_lines for c in sentence]))
        self.characters = list(set([word for sentence in self.sentence_lines for word in sentence]))
        self.num_characters = len(self.characters)
        self.end_token = self.num_characters + 2

    def __len__(self):
        return len(self.sentence_lines)
    
    def encode(self, c):
        # character to int
        if c == '':
            return 0
        elif c == "\n":
            return self.end_token
        else:
            return self.characters.index(c) + 1
    
    def decode(self, i):
        # int to character
        if i == 0:
            return ''
        elif i <= self.num_characters:
            return self.characters[i - 1]
        else:
            return '\n'
        
    def __getitem__(self, i):
        chars = [self.encode(c) for c in self.sentence_lines[i]]
        if self.max_length is None:
            chars_full = [self.start_token] + chars + [self.end_token]*(self.max_length - len(chars))
        else:
            i = np.random.randint(0,len(chars))
            if i == 0:
                chars_full = [self.start_token] + chars[:self.max_length-1]
            else:
                chars_full = chars[i:i+self.max_length]
            if len(chars_full) < self.max_length:
                chars_full = chars_full + [self.end_token]*(self.max_length - len(chars_full))

        # chars_full = [self.start_token] + chars + [self.end_token]*(self.max_length + 2 - len(chars))
        char_tensor = torch.from_numpy(np.array(chars_full).astype(np.int32))
        char_target = torch.from_numpy(np.concatenate([np.array(chars_full[1:]), np.array([self.end_token])]).astype(np.int32))
        return char_tensor, char_target

        # to be or not
        # o be or not 

if __name__ == "__main__":
    dataset = RandomShakespeareSentence()
    for i in range(100,120):
        seq_input, _ = dataset[i]
        seq_string = ''.join(dataset.decode(c) for c in seq_input.detach().numpy())
        print(seq_string)

