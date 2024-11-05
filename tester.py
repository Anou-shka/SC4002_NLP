from datasets import load_dataset
from Embeddings import GloveTokenizer
from normalization import normalize_words

def count_oov_words(tokenizer, texts, use_normalization=False):
    oov = set()

    for text in texts:
        for word in text.split():
            if not use_normalization:
                if word not in tokenizer.word_index:
                    oov.add(word)
                continue

            if word in tokenizer.word_index:
                continue
            subwords = normalize_words(word)
            for sub in subwords:
                if sub not in tokenizer.word_index:
                    oov.add(sub)
    filename = "oov.txt" if use_normalization else "oov_no_norm.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for word in oov:
            f.write(word + "\n")

    return len(oov)

def count_vocab_size(texts, tokenizer):
    unique_words = set()
    vocab_in_glove = set()

    for text in texts:
        for word in text.split():
            unique_words.add(word)
            if word in tokenizer.word_index:
                vocab_in_glove.add(word)

    return len(unique_words), len(vocab_in_glove)

if __name__ == "__main__":
    glove_file_path = "glove.6B.100d.txt"
    tokenizer = GloveTokenizer(glove_file_path)

    dataset = load_dataset("rotten_tomatoes")
    train_dataset = dataset['train']
    train_texts = [example['text'] for example in train_dataset]

    oov_count_no_norm = count_oov_words(tokenizer, train_texts, use_normalization=False)
    oov_count_with_norm = count_oov_words(tokenizer, train_texts, use_normalization=True)
    print(f"OOV count without normalization: {oov_count_no_norm}")
    print(f"OOV count with normalization: {oov_count_with_norm}")

    vocab_size, vocab_in_glove_size = count_vocab_size(train_texts, tokenizer)
    print(f"Vocabulary size in training data: {vocab_size}")
    print(f"Vocabulary size in both training data and GloVe: {vocab_in_glove_size}")
