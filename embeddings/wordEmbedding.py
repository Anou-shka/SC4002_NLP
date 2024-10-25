from datasets import load_dataset
from torchtext.vocab import GloVe
import numpy as np
from normalization import *

dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']
dimension = 100

def tokenize_text(data):
    return [text.split() for text in data]

# Approach 1: Random vectors
def approach1(word):
    global dimension
    return 0, [word], [np.random.normal(size=(dimension,))]

# Approach 2: zero filled vectors
def approach2(word):
    global dimension
    return 0, [word], [np.zeros((dimension,))]

# Approach 3: Simple RegEx
def approach3(count, word, glove, oov2):
    global dimension
    words = normalize_words(word)
    val = []
    for subword in words:
        if subword in glove.stoi:
            val.append(glove.vectors[glove.stoi[subword]].numpy())
        else:
            count += 1
            oov2.add(subword+" : "+word)
            val.append(np.random.normal(size=(dimension,)))
    return count, words, val, oov2

def main():
    train_texts = [example['text'] for example in train_dataset]
    # validation_texts = [example['text'] for example in validation_dataset]
    # test_texts = [example['text'] for example in test_dataset]

    tokenized_texts = tokenize_text(train_texts)

    global dimension
    glove = GloVe(name='6B', dim=dimension)
    vocab = set(word for sentence in tokenized_texts for word in sentence)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    oov = {word for word in vocab if word not in glove.stoi}
    open("oov.txt", "w", encoding='utf-8').write("\n".join(oov))
    print(f"Number of OOV words: {len(oov)}")
    
    embedding = {}
    oov2 = set()
    count = 0
    for word in vocab:
        if word in glove.stoi:
            embedding[word] = glove.vectors[glove.stoi[word]].numpy()
        else:
            count, keys, vals, oov2 = approach3(count, word, glove, oov2)
            for key, val in zip(keys, vals):
                embedding[key] = val
    
    open("oov2.txt", "w", encoding='utf-8').write("\n".join(oov2))
    print(f"Number of OOV words after RegEx: {count}")

if __name__ == '__main__':
    main()