import re
from typing import List

import nltk
import numpy as np


def count_syllabes(word: str) -> int:
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


word_re = re.compile(r"[A-Za-z]+")


def count_words(sentence: str) -> str:
    return word_re.subn("", sentence)[1]


def tokenize_by_sentences(text: str) -> List:
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer.tokenize(text)


def avg_words_per_sentence(sentences: List) -> float:
    return np.mean([count_words(sentence) for sentence in sentences])
