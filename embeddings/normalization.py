import re
import unidecode

def load_contractions(file_path):
    contractions = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                contraction, expansion = line.split(":")
                contraction = contraction.strip()
                expansion = [word.strip() for word in expansion.split(",")]
                contractions[contraction] = expansion
    return contractions

def normalize_words(text): # stupid but works
    words = []
    contractions = load_contractions("contractions.txt")
    for contraction, expansion in contractions.items():
        text = re.sub(rf"\b{contraction}\b", " ".join(expansion), text) # e.g. "I'm" -> "I am"
    text = unidecode.unidecode(text) # e.g. "café" -> "cafe"
    text = re.sub(r"[`\[\]\"]", "", text) # e.g. "`rock[n]roll`" -> "rocknroll"
    text = re.sub(r"(\w+)'s$", r"\1 's", text) # e.g. "John's" -> "John 's"
    text = re.sub(r"(\w+s)'$", r"\1 's", text) # e.g. "dogs'" -> "dogs 's"
    text = re.sub(r"[\']", "", text) # e.g. "rock'n'roll" -> "rocknroll"
    text = re.sub(r"[/-]", " ", text) # e.g. "rock-n-roll" -> "rock n roll"
    text = re.sub(r"([@#&$€£¥])", r" \1 ", text) # e.g. "rock&roll" -> "rock & roll"
    for word in text.split():
        words.append(word)
    return words