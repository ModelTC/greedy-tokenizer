# Copyright 2023 Chielo Newctle <ChieloNewctle@gmail.com>
# Copyright 2023 ModelTC Team
#
# Licensed under either of
# - Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# - MIT license: https://opensource.org/licenses/MIT
# at your option.

import os
from timeit import timeit
from typing import Sequence

import datasets


def load_dataset() -> Sequence[str]:
    name = os.getenv("DATASET", "Trelis/tiny-shakespeare")
    split_name = os.getenv("SPLIT", "train")
    d = datasets.load_dataset(name, split=split_name)  # pyright: ignore
    column_name = os.getenv("COLUMN", d.column_names[0])  # pyright: ignore
    return d[column_name]  # pyright: ignore


DATASET = load_dataset()
NUMBER = int(os.getenv("NUMBER", "50"))


def do_benchmarks(func_seq):
    print("running benchmarks...")
    for func in func_seq:
        print(f"{func.__name__:20} {timeit(func, number=NUMBER):>6.2f}")


def bench_tokenizer(tokenizer):
    def tokenize():
        for seq in DATASET:
            tokenizer.tokenize(seq)

    def encode():
        for seq in DATASET:
            tokenizer.encode(seq)

    def encode_and_decode():
        for seq in DATASET:
            tokenizer.decode(tokenizer.encode(seq))

    do_benchmarks([tokenize, encode, encode_and_decode])
