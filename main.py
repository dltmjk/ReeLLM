import re
import tiktoken
from SimpleTokenizer import SimpleTokenizer
import importlib

# Extracting Raw Text
with open("ReeLRBWittgenstein.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))

# Preprocessing for Tokenization
preprocessed = re.split(r'([,.:;?_!"()\'-]|\s)', raw_text)
preprocessed = [
 item.strip() for item in preprocessed if item.strip()
 ]
print(preprocessed)

# Vocabulary Size
all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# Convert into token IDs
tokenizer = tiktoken.get_encoding("gpt2")

text = "Wilson isn’t much of a scientist: at one point he muses that the invention of the steam engine depended on the discovery that water is H2O, which is bizarre both conceptually and chronologically. But as far as he is concerned Darwin wasn’t a scientist either"

integers = tokenizer.encode(text)
print(integers)

strings = tokenizer.decode(integers)
print(strings)