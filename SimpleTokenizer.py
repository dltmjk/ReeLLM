import re
class SimpleTokenizer:
    def __init__(self, vocab):
        self.StringToInt = vocab
        self.IntToString = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\'-]|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.StringToInt
                        else "<|unk|>" for item in preprocessed]
        ids = [self.StringToInt[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.IntToString[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'-])', r'\1', text)
        return text

