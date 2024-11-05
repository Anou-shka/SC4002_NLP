import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from normalization import normalize_words
import torch.nn as nn

class GloveTokenizer:
    def __init__(self, glove_file_path, pad_token="<PAD>", unk_token="<UNK>"):
        # Load GloVe embeddings from file
        self.embeddings_index = self._load_glove_embeddings(glove_file_path)
        self.pad_token = pad_token
        self.pad_token_id = 0
        self.unk_token = unk_token
        self.unk_token_id = 1

        # Create word-to-index and index-to-word dictionaries
        self.word_index = {word: idx for idx, word in enumerate(self.embeddings_index.keys(), start=2)}
        self.word_index[self.pad_token] = self.pad_token_id
        self.word_index[self.unk_token] = self.unk_token_id
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    def _load_glove_embeddings(self, glove_file_path):
        # Load the GloVe embeddings from file into a dictionary
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        return embeddings_index

    def _tokenize_with_subwords(self, word):
        # If word exists in vocabulary, return its index
        if word in self.word_index:
            return [self.word_index[word]]

        # Otherwise, tokenize word into subwords and try to match each subword
        subword_tokens = []
        subwords = normalize_words(word)
        for subword in subwords:
            if subword in self.word_index:
                subword_tokens.append(self.word_index[subword])

        # Return indices of matched subwords, or UNK token if no subwords matched
        if subword_tokens:
            return subword_tokens
        else:
            return [self.unk_token_id]

    def encode(self, texts, max_length=None, return_tensors="list"):
        # Convert single string to list for consistent handling
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize and encode each text in batch
        all_input_ids = []
        for text in texts:
            input_ids = []
            for word in text.split():
                input_ids.extend(self._tokenize_with_subwords(word))
            all_input_ids.append(input_ids)

        # Pad sequences and create attention masks
        input_ids_padded = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in all_input_ids],
            batch_first=True,
            padding_value=self.pad_token_id
        )

        attention_mask = (input_ids_padded != self.pad_token_id).long()

        # Trim/pad to max_length if specified
        if max_length:
            input_ids_padded = input_ids_padded[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

            if input_ids_padded.shape[1] < max_length:
                pad_size = max_length - input_ids_padded.shape[1]
                padding = torch.full((input_ids_padded.shape[0], pad_size), self.pad_token_id, dtype=torch.long)
                input_ids_padded = torch.cat([input_ids_padded, padding], dim=1)
                
                mask_padding = torch.zeros((attention_mask.shape[0], pad_size), dtype=torch.long)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        # Return in specified format
        if return_tensors == "pt":
            return {"input_ids": input_ids_padded, "attention_mask": attention_mask}
        else:
            return {"input_ids": input_ids_padded.tolist(), "attention_mask": attention_mask.tolist()}

    def decode(self, token_ids_batch, skip_special_tokens=False):
        # Convert tensor to list for consistent handling
        if isinstance(token_ids_batch, torch.Tensor):
            if token_ids_batch.dim() != 2:
                raise ValueError("Input tensor must be 2-dimensional.")
            token_ids_batch = token_ids_batch.tolist()

        # Ensure batch format
        if isinstance(token_ids_batch[0], int):
            token_ids_batch = [token_ids_batch]

        # Decode each sequence in batch
        all_texts = []
        for token_ids in token_ids_batch:
            words = []
            for idx in token_ids:
                word = self.index_word.get(idx, self.unk_token)
                if skip_special_tokens and word == self.pad_token:
                    continue
                words.append(word)
            all_texts.append(" ".join(words))

        # Return single or list of decoded texts
        return all_texts if len(all_texts) > 1 else all_texts[0]

class GloveTokenizerNoSub(GloveTokenizer):
    def __init__(self, glove_file_path, pad_token="<PAD>", unk_token="<UNK>"):
        super().__init__(glove_file_path, pad_token, unk_token)

    def _tokenize_with_subwords(self, word):
        if word in self.word_index:
            return [self.word_index[word]]
        else:
            return [self.unk_token_id]

class GloveEmbedding(nn.Module):
    def __init__(self, glove_file_path, embedding_dim=100, trainable=False, pad_token="<PAD>", unk_token="<UNK>"):
        super(GloveEmbedding, self).__init__()
        # Initialize vocabulary and embedding matrix with GloVe embeddings
        self.word_index, embedding_matrix = self._load_glove_embeddings(glove_file_path, embedding_dim, pad_token, unk_token)
        
        # Set up embedding layer with GloVe weights
        vocab_size = embedding_matrix.shape[0]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = trainable

    def _load_glove_embeddings(self, glove_file_path, embedding_dim, pad_token, unk_token):
        # Load GloVe embeddings and build vocabulary and embedding matrix
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        
        # Create word index, and initialize embedding matrix
        word_index = {word: idx for idx, word in enumerate(embeddings_index.keys(), start=2)}
        word_index[pad_token] = 0
        word_index[unk_token] = 1

        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            elif word == unk_token:
                embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))
            elif word == pad_token:
                embedding_matrix[idx] = np.zeros(embedding_dim)
        
        return word_index, embedding_matrix

    def forward(self, x):
        # Perform embedding lookup
        return self.embedding(x)

def test(tokenizer):
    embedding_layer = GloveEmbedding(glove_file_path, embedding_dim=100)

    # Example batch of texts
    texts = ["hello world this is <UNK> test",
            "another example sentence for testing",
            "batch processing with <PAD> and <UNK> tokens",
            "Check the price of the new gadget ($199) at 'Tech-Store', " \
        "and don't forget to use the discount code 'SAVE20' for 20% off on your next purchase! " \
        "For more info, call #123 or visit www.tech-store.com & sign up."]

    # Encode batch with tokenizer, returning list format
    encoded_list = tokenizer.encode(texts, max_length=60, return_tensors="list")
    print("Encoded Batch (List):", encoded_list)

    # Encode batch with tokenizer, returning torch tensor format
    encoded_tensor = tokenizer.encode(texts, max_length=60, return_tensors="pt")
    print("Encoded Batch (Torch):", encoded_tensor)
    print("Shape:", encoded_tensor["input_ids"].shape, encoded_tensor["attention_mask"].shape)

    # Decode batch with tokenizer
    decoded_texts = tokenizer.decode(encoded_tensor["input_ids"])
    print("Decoded Batch (Tensor):", decoded_texts)
    
    # Decode list format
    decoded_texts = tokenizer.decode(encoded_list["input_ids"])
    print("Decoded Batch (List):", decoded_texts)
    
    # Pass encoded input_ids through embedding layer
    embedded_output = embedding_layer(encoded_tensor["input_ids"])
    print("Embedded Output Shape (Batch):", embedded_output.shape)
    print("Embedded Output Shape (Batch):", embedded_output)
    
    # Verify alignment of embeddings
    test_word = "text"  # Word to check
    if test_word in tokenizer.word_index:
        test_index = tokenizer.word_index[test_word]
        
        # Retrieve GloVe vector
        glove_vector = torch.tensor(tokenizer.embeddings_index[test_word], dtype=torch.float32)
        
        # Retrieve nn.Embedding vector
        embedding_vector = embedding_layer.embedding.weight[test_index]
        
        # Print and compare vectors
        print(f"GloVe vector for '{test_word}':", glove_vector)
        print(f"Embedding vector for '{test_word}':", embedding_vector)
        
        # Check if vectors align
        if torch.allclose(glove_vector, embedding_vector, atol=1e-6):
            print(f"The embedding for '{test_word}' is correctly aligned with the original GloVe vector.")
        else:
            print(f"The embedding for '{test_word}' is NOT aligned with the original GloVe vector.")
    else:
        print(f"The word '{test_word}' is not in the tokenizer's vocabulary.")



if __name__ == "__main__":
    # Test script
    glove_file_path = "glove.6B.100d.txt"

    # Initialize Tokenizer and Embedding layer
    tokenizer = GloveTokenizer(glove_file_path)
    test(tokenizer)
    tokenizer_no_sub = GloveTokenizerNoSub(glove_file_path)
    test(tokenizer_no_sub)