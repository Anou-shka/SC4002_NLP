import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

class GloveTokenizer:
    def __init__(self, glove_file_path, pad_token="<PAD>", unk_token="<UNK>"):
        # 加载 GloVe 词向量并创建词典
        self.embeddings_index = self._load_glove_embeddings(glove_file_path)
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # 初始化词汇索引表，并确保特殊标记
        self.word_index = {word: idx for idx, word in enumerate(self.embeddings_index.keys(), start=2)}
        self.word_index[self.pad_token] = 0
        self.word_index[self.unk_token] = 1
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    def _load_glove_embeddings(self, glove_file_path):
        # 从文件中加载 GloVe 向量
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        return embeddings_index

    def encode(self, texts, max_length=None, return_tensors="list"):
        # 检查输入是否是单一字符串，将其转换为列表以处理批量输入
        if isinstance(texts, str):
            texts = [texts]

        # 将文本转为 token_ids 列表
        all_input_ids = [[self.word_index.get(word, self.word_index[self.unk_token]) for word in text.split()] for text in texts]

        # 使用 pad_sequence 填充到相同长度
        input_ids_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in all_input_ids],
                                        batch_first=True,
                                        padding_value=self.word_index[self.pad_token])

        # 自动生成 attention_mask，注意此处返回一个二维张量
        attention_mask = (input_ids_padded != self.word_index[self.pad_token]).long()

        # 若指定了 max_length，进行截断或再次填充
        if max_length:
            input_ids_padded = input_ids_padded[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

            if input_ids_padded.shape[1] < max_length:
                pad_size = max_length - input_ids_padded.shape[1]
                padding = torch.full((input_ids_padded.shape[0], pad_size), self.word_index[self.pad_token], dtype=torch.long)
                input_ids_padded = torch.cat([input_ids_padded, padding], dim=1)
                
                mask_padding = torch.zeros((attention_mask.shape[0], pad_size), dtype=torch.long)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        # 返回格式改为二维张量
        if return_tensors == "pt":
            return {"input_ids": input_ids_padded, "attention_mask": attention_mask}
        else:
            return {"input_ids": input_ids_padded.tolist(), "attention_mask": attention_mask.tolist()}

    def decode(self, token_ids_batch, skip_special_tokens=False):
        # 检查输入是否为二维 Tensor
        if isinstance(token_ids_batch, torch.Tensor):
            if token_ids_batch.dim() != 2:
                raise ValueError("Input tensor must be 2-dimensional.")
            token_ids_batch = token_ids_batch.tolist()  # 转换为列表以便处理

        # 检查输入是否为单个句子，将其转换为列表以处理批量输入
        if isinstance(token_ids_batch[0], int):
            token_ids_batch = [token_ids_batch]

        all_texts = []
        for token_ids in token_ids_batch:
            words = []
            for idx in token_ids:
                # 确保 idx 是整数，进行映射
                word = self.index_word.get(idx, self.unk_token)
                # 跳过填充标记
                if skip_special_tokens and word == self.pad_token:
                    continue
                words.append(word)
            all_texts.append(" ".join(words))

        return all_texts if len(all_texts) > 1 else all_texts[0]




class GloveEmbedding(nn.Module):
    def __init__(self, glove_file_path, embedding_dim=100, trainable=False, pad_token="<PAD>", unk_token="<UNK>"):
        super(GloveEmbedding, self).__init__()
        self.word_index, embedding_matrix = self._load_glove_embeddings(glove_file_path, embedding_dim, pad_token, unk_token)
        
        # 创建嵌入层
        vocab_size = embedding_matrix.shape[0]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = trainable  # 根据trainable控制是否可训练

    def _load_glove_embeddings(self, glove_file_path, embedding_dim, pad_token, unk_token):
        # 加载词向量并构建词-索引字典和嵌入矩阵
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        
        # 初始化词汇索引表，添加特殊标记
        word_index = {word: idx for idx, word in enumerate(embeddings_index.keys(), start=2)}
        word_index[pad_token] = 0
        word_index[unk_token] = 1

        # 构建嵌入矩阵
        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            elif word == unk_token:
                embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))  # 随机初始化UNK
            elif word == pad_token:
                embedding_matrix[idx] = np.zeros(embedding_dim)  # <PAD>为零向量
        
        return word_index, embedding_matrix

    def forward(self, x):
        return self.embedding(x)


if __name__ == "__main__":
    # 测试代码
    glove_file_path = "C:\\Users\\Administrator\\Desktop\\glove\\glove.6B.100d.txt"

    # 初始化 Tokenizer 和 Embedding
    tokenizer = GloveTokenizer(glove_file_path)
    embedding_layer = GloveEmbedding(glove_file_path, embedding_dim=100)

    # 示例文本批次
    texts = ["hello world this is <UNK> test",
            "another example sentence for testing",
            "batch processing with <PAD> and <UNK> tokens"]

    # 使用 Tokenizer 编码文本批次，返回list格式
    encoded_list = tokenizer.encode(texts, max_length=12, return_tensors="list")
    print("Encoded Batch (List):", encoded_list)

    # 使用 Tokenizer 编码文本批次，返回pt格式
    encoded_tensor = tokenizer.encode(texts, max_length=12, return_tensors="pt")
    print("Encoded Batch (Torch):", encoded_tensor)

    # 使用 Tokenizer 解码批次
    decoded_texts = tokenizer.decode(encoded_tensor["input_ids"])
    print("Decoded Batch (Tensor):", decoded_texts)
    
    # 使用 Tokenizer 解码
    decoded_texts = tokenizer.decode(encoded_list["input_ids"])
    print("Decoded Batch (List):", decoded_texts)
    
    # 将编码后的 input_ids 输入到 Embedding 层
    embedded_output = embedding_layer(encoded_tensor["input_ids"])
    print("Embedded Output Shape (Batch):", embedded_output.shape)
    print("Embedded Output Shape (Batch):", embedded_output)
    
    # 验证嵌入是否对齐
    test_word = "text"  # 要测试的单词
    if test_word in tokenizer.word_index:
        test_index = tokenizer.word_index[test_word]
        
        # 从 GloVe 原始嵌入中获取该单词的向量
        glove_vector = torch.tensor(tokenizer.embeddings_index[test_word], dtype=torch.float32)
        
        # 从 nn.Embedding 中获取该单词的向量
        embedding_vector = embedding_layer.embedding.weight[test_index]
        
        # 打印并比较两个向量
        print(f"GloVe vector for '{test_word}':", glove_vector)
        print(f"Embedding vector for '{test_word}':", embedding_vector)
        
        # 检查是否对齐
        if torch.allclose(glove_vector, embedding_vector, atol=1e-6):
            print(f"The embedding for '{test_word}' is correctly aligned with the original GloVe vector.")
        else:
            print(f"The embedding for '{test_word}' is NOT aligned with the original GloVe vector.")
    else:
        print(f"The word '{test_word}' is not in the tokenizer's vocabulary.")
