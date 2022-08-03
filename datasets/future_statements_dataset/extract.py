#!/usr/bin/env python

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("future_dataset.csv", sep='|')

    with open("X_train.csv", "w") as f:
        f.write("statement\n")
        for row in df["statement"]:
            f.write('"' + row + '"\n')

    with open("y_train.csv", "w") as f:
        f.write("future\n")
        for row in df["future"]:
            f.write("0\n" if row == "no" else "1\n")
