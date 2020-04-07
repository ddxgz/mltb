import random

import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer


def text_random_crop(rec, crop_by: str = 'word', crop_ratio: float = 0.1):
    if crop_by == 'word':
        sents = nltk.word_tokenize(rec)
    elif crop_by == 'sentence':
        sents = nltk.sent_tokenize(rec)
    else:
        sents = rec
    size = len(sents)
    chop_size = size // (1 / crop_ratio)
    chop_offset = random.randint(0, int(chop_size))
    sents_chop = sents[chop_offset:size - chop_offset - 1]

    d = TreebankWordDetokenizer()
    return d.detokenize(sents_chop)
