#!/usr/bin/env python3

import multiprocessing
import nltk
import re
from glob import glob


def process(filename) -> None:
    with open(filename) as ff:
        data = ff.read()
    filename_out = filename[:-4] + "_extracted.txt"

    with open(filename_out, "w") as ff:
        tokens = nltk.sent_tokenize(data)
        for t in tokens:
            if not keep_token(t): continue
            t = preprocess_token(t)
            ff.write("%s|NULL|Yes|UN General Debates Corpus\n" % t)

    print(filename + " processed.")


def keep_token(token) -> bool:
    future_matches = []
    future_keywords = ["will",
                       "'ll",
                       "going to",
                       ]
    blacklisted = ["going to school",
                   "going to the",
                   "hope",
                   "hoped",
                   "hoping",
                   ]

    with open("./english_verbs", mode="r") as ff:
        english_verbs = ff.readlines()
    for future_keyword in future_keywords:
        for english_verb in english_verbs:
            future_matches.append(future_keyword.strip() +
                                  " " +
                                  english_verb.strip())

    if not any(blck in token for blck in blacklisted) and \
       any(match in token for match in future_matches) and \
       "“" not in token and \
       "”" not in token and \
       "," not in token and \
       "?" not in token and \
       "—" not in token and \
       "\"" not in token:
        return True

    return False


def preprocess_token(token) -> str:
    token = re.sub("\n", r" ", token)
    token = token.strip()
    return token


if __name__ == "__main__":
    nltk.download("punkt")
    p = multiprocessing.Pool()

    for f in glob("*.txt"):
        p.apply_async(process, args=(f,))

    p.close()
    p.join()
