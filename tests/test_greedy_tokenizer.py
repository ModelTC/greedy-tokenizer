# Copyright 2023 Chielo Newctle <ChieloNewctle@gmail.com>
# Copyright 2023 ModelTC Team
#
# Licensed under either of
# - Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# - MIT license: https://opensource.org/licenses/MIT
# at your option.

import os
from tempfile import TemporaryDirectory
from typing import Sequence

import datasets
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from greedy_tokenizer import GreedyTokenizer, UTF8Buffer


def mock_other_tokenizer(**kwargs):
    kwargs.setdefault("add_bos_token", False)
    kwargs.setdefault("add_eos_token", False)
    name = os.getenv("SP_TOKENIZER") or "codellama/CodeLlama-7b-Instruct-hf"
    old_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(name)
    return old_tokenizer, GreedyTokenizer.mock_tokenizer(old_tokenizer, **kwargs)


def load_dataset() -> Sequence[str]:
    name = os.getenv("DATASET") or "Trelis/tiny-shakespeare"
    split_name = os.getenv("SPLIT") or "train"
    d = datasets.load_dataset(name, split=split_name)  # pyright: ignore
    column_name = os.getenv("COLUMN") or d.column_names[0]  # pyright: ignore
    return d[column_name]  # pyright: ignore


def test_store_and_load():
    _, tokenizer = mock_other_tokenizer()

    with TemporaryDirectory() as dir_name:
        paths = tokenizer.save_pretrained(dir_name)
        vocab_path = next(
            i
            for i in paths
            if i.endswith(GreedyTokenizer.GREEDY_TOKENIZER_VOCAB_FILE_NAME)
        )

        auto_tokenizer = AutoTokenizer.from_pretrained(dir_name, trust_remote_code=True)
        from_pretrained = GreedyTokenizer.from_pretrained(dir_name)
        from_vocab = GreedyTokenizer(vocab_path=vocab_path)

    assert type(auto_tokenizer).__module__ != "greedy_tokenizer"
    assert type(auto_tokenizer).__name__ == "GreedyTokenizer"

    assert type(from_pretrained).__module__ == "greedy_tokenizer"
    assert type(from_pretrained).__name__ == "GreedyTokenizer"

    assert auto_tokenizer.vocab == tokenizer.vocab
    assert from_pretrained.vocab == tokenizer.vocab
    assert from_vocab.vocab == tokenizer.vocab


def test_unk_token():
    t = GreedyTokenizer(
        vocab=["<unk>", "Hello", "world", " ", ",", ".", "!"],
        unk_token="<unk>",
    )

    assert t.tokenize("Hello, world!") == ["Hello", ",", " ", "world", "!"]
    assert t.tokenize("Goodbye, world.") == ["<unk>", ",", " ", "world", "."]

    assert t.convert_tokens_to_string(["<unk>", "<unk>"]) == "<unk>"


def test_decode_invalid_utf8():
    t = GreedyTokenizer(
        vocab=["<unk>", "<0xe4>", "<0xbd>", "<0xff>", "hello", "<0x123>"],
        unk_token="<unk>",
    )

    def case(seq):
        assert t.convert_tokens_to_string([*seq]) == "<unk>"
        assert t.convert_tokens_to_string(["hello", *seq]) == "hello<unk>"
        assert t.convert_tokens_to_string([*seq, "hello"]) == "<unk>hello"
        assert t.convert_tokens_to_string(["hello", *seq, "hello"]) == "hello<unk>hello"

    case(["<0xe4>", "<0xbd>"])
    case(["<0xff>"])
    case(["<0xe4>", "<0xff>"])
    case(["<0xff>", "<0xff>", "<0xe4>", "<0xbd>", "<0xff>"])

    assert t.convert_tokens_to_string(["<0x123>"]) == "<0x123>"


def test_empty_token():
    t = GreedyTokenizer(vocab=["Hello", "", "world", " ", ",", "", ".", "!"])

    assert t.tokenize("Hello, world!") == ["Hello", ",", " ", "world", "!"]


def test_tokenize_dataset():
    sp_tokenizer, gt_tokenizer = mock_other_tokenizer()

    assert gt_tokenizer.vocab_size == sp_tokenizer.vocab_size

    tot_num_sp, tot_num_gt = 0, 0

    for item in load_dataset():
        num_sp = len(sp_tokenizer.tokenize(item))

        res = gt_tokenizer.tokenize(item)
        num_gt = len(res)

        exceeded_ratio = (num_gt - num_sp) / num_sp

        assert gt_tokenizer.convert_tokens_to_string(res) == item
        assert exceeded_ratio < 1.6

        token_ids = gt_tokenizer.convert_tokens_to_ids(res)
        assert gt_tokenizer.convert_ids_to_tokens(token_ids) == res
        for token in res:
            token_id = gt_tokenizer.convert_tokens_to_ids(token)
            assert gt_tokenizer.convert_ids_to_tokens(token_id) == token

        tot_num_sp += num_sp
        tot_num_gt += num_gt

    exceeded_ratio = (tot_num_gt - tot_num_sp) / tot_num_sp
    print(f"{tot_num_sp=} {tot_num_gt=} {exceeded_ratio * 100.0=:.2}%")


def test_exceptions():
    with pytest.raises(TypeError):
        GreedyTokenizer()

    with pytest.raises(UnicodeDecodeError):
        t = GreedyTokenizer(vocab=["<0xff>"])
        t.convert_tokens_to_string(["<0xff>"])

    buffer = UTF8Buffer("<unk>")
    buffer.push_byte(-1)
    buffer.push_byte(0x100)
    assert buffer.pop_chars() == "<unk>"

    with TemporaryDirectory() as dir_name:
        t = GreedyTokenizer(vocab=[])
        paths = t.save_pretrained(dir_name, filename_prefix="t")
        assert any(
            "t-" + GreedyTokenizer.GREEDY_TOKENIZER_VOCAB_FILE_NAME in p for p in paths
        )


def test_bos_eos():
    def get_tokenizer(**kwargs):
        def sub_bos_and_eos(token_id: int, token: str, _: bool) -> str:
            return {1: "<s>", 2: "</s>"}.get(token_id, token)

        _, tokenizer = mock_other_tokenizer(proc_token=sub_bos_and_eos, **kwargs)

        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = "<s>"

        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "</s>"

        return tokenizer

    tokenizer = get_tokenizer(add_bos_token=True, add_eos_token=False)

    res = tokenizer.encode("xyz")
    assert res.count(tokenizer.bos_token_id) == 1  # pyright: ignore
    assert res.count(tokenizer.eos_token_id) == 0  # pyright: ignore
    res = tokenizer.encode("uvw", "ijk")
    assert res.count(tokenizer.bos_token_id) == 2  # pyright: ignore
    assert res.count(tokenizer.eos_token_id) == 0  # pyright: ignore

    tokenizer = get_tokenizer(add_bos_token=True, add_eos_token=True)

    res = tokenizer.encode("xyz")
    assert res.count(tokenizer.bos_token_id) == 1  # pyright: ignore
    assert res.count(tokenizer.eos_token_id) == 1  # pyright: ignore
    res = tokenizer.encode("uvw", "ijk")
    assert res.count(tokenizer.bos_token_id) == 2  # pyright: ignore
    assert res.count(tokenizer.eos_token_id) == 2  # pyright: ignore

    tokenizer = get_tokenizer(add_bos_token=False, add_eos_token=True)

    res = tokenizer.encode("xyz")
    assert res.count(tokenizer.bos_token_id) == 0  # pyright: ignore
    assert res.count(tokenizer.eos_token_id) == 1  # pyright: ignore
    res = tokenizer.encode("uvw", "ijk")
    assert res.count(tokenizer.bos_token_id) == 0  # pyright: ignore
    assert res.count(tokenizer.eos_token_id) == 2  # pyright: ignore


def test_decode_special_tokens():
    _, tokenizer = mock_other_tokenizer(add_bos_token=False, add_eos_token=False)
    tokenizer._add_tokens(["<wow>", "<yeah>"], special_tokens=True)

    def proc(s):
        return tokenizer.decode(tokenizer.encode(s))

    assert proc("ab<wow>cd<yeah>ef") == "ab<wow>cd<yeah>ef"
    assert proc("\n \t<wow> <yeah>\t \n") == "\n \t<wow> <yeah>\t \n"
