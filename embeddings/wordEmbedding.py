import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from normalization import normalize_words
import random

random.seed(19260817)
np.random.seed(19260817)

class CustomTokenizer:
    def __init__(self, dimension=100, glove_path="glove.6B.100d.txt"):
        self.dimension = dimension
        self.tokens = {}
        self.token_to_id = {}
        self.embeddings = [np.zeros(self.dimension)]  # PAD token embedding
        self.oov2 = set()
        self.pad_token_id = 0  # Set a special token ID for padding
        self.unk_token_id = 1  # Set a special token ID for unknown tokens
        
        # Initialize UNK token with random embedding
        self.embeddings.append(np.random.normal(0, 0.1, (self.dimension,)))
        self.tokens["<UNK>"] = self.unk_token_id
        self.token_to_id[self.unk_token_id] = "<UNK>"

        # Load GloVe embeddings from file
        self.glove = self.load_glove_embeddings(glove_path)

    def load_glove_embeddings(self, glove_path):
        glove_embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                glove_embeddings[word] = vector
        return glove_embeddings

    def build_vocab(self, data):
        # First pass: collect all normalized tokens
        all_normalized_tokens = set()
        for text in data:
            words = text.split()
            for word in words:
                normalized = normalize_words(word)
                all_normalized_tokens.update(normalized)

        # Sort for deterministic ordering
        vocab = sorted(all_normalized_tokens)
        self.oov = {word for word in vocab if word not in self.glove}
        count = 2  # Start from 2 as 0 and 1 are reserved for PAD and UNK
        for word in vocab:
            if word in self.glove:
                self.tokens[word] = count
                self.token_to_id[count] = word
                self.embeddings.append(self.glove[word])
                count += 1
            else:
                # For tokens not in GloVe, initialize with random embedding
                self.tokens[word] = count
                self.token_to_id[count] = word
                self.embeddings.append(np.random.normal(0, 0.1, (self.dimension,)))
                count += 1

        self.embeddings = np.array(self.embeddings)

    def tokenize(self, text):
        subwordseq = []
        words = text.split()
        for word in words:
            normalized_tokens = normalize_words(word)
            token_found = False
            for token in normalized_tokens:
                if token in self.tokens:
                    subwordseq.append(self.tokens[token])
                    token_found = True
                else:
                    subwordseq.append(self.unk_token_id)
            # If no tokens were added (empty normalized_tokens), add UNK
            if not normalized_tokens:
                subwordseq.append(self.unk_token_id)

        return subwordseq

    def encode(self, text, return_tensor=None, max_length=None):
        input_ids = self.tokenize(text)

        if max_length is not None:
            input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)  
        padding_length = (max_length - len(input_ids)) if max_length else 0
        if padding_length > 0:
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        if return_tensor == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.int64),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.int64)
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

    def decode(self, token_ids):
        return " ".join(self.token_to_id[id] for id in token_ids if id in self.token_to_id)

    def convert_tokens_to_ids(self, tokens):
        return [self.tokens.get(token, self.unk_token_id) for token in tokens]

    def embedding(self, token):
        return self.embeddings[token]

# Collater is used for batch processing to generate padded data in bulk
class Collater:
    def __init__(self, tokenizer: CustomTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: list, max_length=None):
        input_ids = [instance["input_ids"].clone().detach().to(torch.int64) for instance in instances]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).type(torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

if __name__ == '__main__':
    dataset = load_dataset("rotten_tomatoes")
    train_dataset = dataset['train']
    train_texts = [example['text'] for example in train_dataset]

    tokenizer = CustomTokenizer(dimension=100, glove_path="glove.6B.100d.txt")
    tokenizer.build_vocab(train_texts)

    encoding = tokenizer.encode("Check the price of the new gadget ($199) at 'Tech-Store', " \
        "and don't forget to use the discount code 'SAVE20' for 20% off on your next purchase! " \
        "For more info, call #123 or visit www.tech-store.com & sign up.", return_tensor="pt", max_length=60)
    print("Encoding with padding:", encoding)

    decoded_text = tokenizer.decode(encoding['input_ids'].tolist())
    print("Decoded text:", decoded_text)

    tokens = ["check", "the", "price", "of", "the", "new"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Converted token IDs:", token_ids)

    print("Embedding of token ID 114:", tokenizer.embedding(114))

    collater = Collater(tokenizer)
    batched_data = collater([encoding], max_length=60)
    print("Batched data:", batched_data)
