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
    text = unidecode.unidecode(text) # e.g. "café" -> "cafe"
    text = text.lower()
    for contraction, expansion in contractions.items():
        text = re.sub(rf"\b{contraction}\b", " ".join(expansion), text) # e.g. "I'm" -> "I am"
    text = re.sub(r"[`\[\]\"]", "", text) # e.g. "`rock[n]roll`" -> "rocknroll"
    text = re.sub(r"'s(?=[^a-zA-Z]|$)", r" 's ", text) # e.g. "John's" -> "John 's"
    text = re.sub(r"s'(?=[^a-zA-Z]|$)", r"s 's ", text) # e.g. "dogs'" -> "dogs 's"
    text = re.sub(r"[/-]", " ", text) # e.g. "rock-n-roll" -> "rock n roll"
    text = re.sub(r"([@#&%+:,.?!$€£¥\(\)])", r" \1 ", text) # e.g. "rock&roll" -> "rock & roll"
    text = re.sub(r"(?<!s)'(?!s\b\s|s$)", "", text) # e.g. "rock'n'roll" -> "rocknroll"
    for word in text.split():
        words.append(word)
    return words

if __name__ == "__main__":
    text = "Check the price of the new gadget ($199) at 'Tech-Store', " \
           "and don't forget to use the discount code 'SAVE20' for 20% off on your next purchase! " \
           "For more info, call #123 or visit www.tech-store.com & sign up."
    print(text)
    print(normalize_words(text))
