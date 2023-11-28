# Copyright 2023 Chielo Newctle <ChieloNewctle@gmail.com>
# Copyright 2023 ModelTC Team
#
# Licensed under either of
# - Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# - MIT license: https://opensource.org/licenses/MIT
# at your option.

import os
import json
from tempfile import TemporaryDirectory
from typing import Sequence

import datasets
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from greedy_tokenizer import GreedyTokenizer, UTF8Buffer


try:
    from greedy_tokenizer import GreedyTokenizerFast

    pytestmark = pytest.mark.parametrize(
        "factory", [GreedyTokenizer, GreedyTokenizerFast]
    )
except ImportError:
    GreedyTokenizerFast = None
    pytestmark = pytest.mark.parametrize("factory", [GreedyTokenizer])


FAST_BYTE_FALLBACK = "�"


def mock_other_tokenizer(factory, **kwargs):
    kwargs.setdefault("add_bos_token", False)
    kwargs.setdefault("add_eos_token", False)
    name = os.getenv("OLD_TOKENIZER", "codellama/CodeLlama-7b-Instruct-hf")
    old_kwargs = json.loads(os.getenv("OLD_TOKENIZER_KWARGS", "{}"))
    old_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        name, **old_kwargs
    )
    return old_tokenizer, factory.mock_tokenizer(old_tokenizer, **kwargs)


def load_dataset() -> Sequence[str]:
    name = os.getenv("DATASET", "Trelis/tiny-shakespeare")
    split_name = os.getenv("SPLIT", "train")
    d = datasets.load_dataset(name, split=split_name)  # pyright: ignore
    column_name = os.getenv("COLUMN", d.column_names[0])  # pyright: ignore
    return d[column_name]  # pyright: ignore


def test_store_and_load(factory):
    _, tokenizer = mock_other_tokenizer(factory)

    with TemporaryDirectory() as dir_name:
        paths = tokenizer.save_pretrained(dir_name)
        vocab_path = next(
            i for i in paths if i.endswith(factory.GREEDY_TOKENIZER_VOCAB_FILE_NAME)
        )

        auto_tokenizer = AutoTokenizer.from_pretrained(
            dir_name,
            trust_remote_code=True,
            use_fast=factory != GreedyTokenizer,
        )
        from_pretrained = factory.from_pretrained(dir_name)
        from_vocab = factory(vocab_path=vocab_path)

    assert type(auto_tokenizer).__module__ != "greedy_tokenizer"
    assert type(auto_tokenizer).__name__ == factory.__name__

    assert type(from_pretrained).__module__ == "greedy_tokenizer"
    assert type(from_pretrained).__name__ == factory.__name__

    assert auto_tokenizer.vocab == tokenizer.vocab
    assert from_pretrained.vocab == tokenizer.vocab
    assert from_vocab.vocab == tokenizer.vocab


def test_unk_token(factory):
    t = factory(
        vocab=["<unk>", "Hello", "world", " ", ",", ".", "!"],
        unk_token="<unk>",
    )

    assert t.tokenize("Hello, world!") == ["Hello", ",", " ", "world", "!"]
    assert t.tokenize("Goodbye, world.") == ["<unk>", ",", " ", "world", "."]

    if isinstance(t, GreedyTokenizer):
        assert t.convert_tokens_to_string(["<unk>", "<unk>"]) == "<unk>"


def test_decode_invalid_utf8(factory):
    t = factory(
        vocab=["<unk>", "<0xe4>", "<0xbd>", "<0xff>", "hello", "<0x123>"],
        unk_token="<unk>",
    )

    def case_slow(seq):
        assert t.convert_tokens_to_string([*seq]) == "<unk>"
        assert t.convert_tokens_to_string(["hello", *seq]) == "hello<unk>"
        assert t.convert_tokens_to_string([*seq, "hello"]) == "<unk>hello"
        assert t.convert_tokens_to_string(["hello", *seq, "hello"]) == "hello<unk>hello"

    def case_fast(seq):
        assert t.convert_tokens_to_string([*seq]) == FAST_BYTE_FALLBACK * len(seq)

    case = case_slow if factory == GreedyTokenizer else case_fast

    case(["<0xe4>", "<0xbd>"])
    case(["<0xff>"])
    case(["<0xe4>", "<0xff>"])
    case(["<0xff>", "<0xff>", "<0xe4>", "<0xbd>", "<0xff>"])

    assert t.convert_tokens_to_string(["<0x123>"]) == "<0x123>"


def test_empty_token(factory):
    t = factory(vocab=["Hello", "", "world", " ", ",", "", ".", "!"])

    assert t.tokenize("Hello, world!") == ["Hello", ",", " ", "world", "!"]


def test_tokenize_dataset(factory):
    sp_tokenizer, gt_tokenizer = mock_other_tokenizer(factory)

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


def test_encode_and_decode_dataset(factory):
    _, gt_tokenizer = mock_other_tokenizer(
        factory, add_bos_token=False, add_eos_token=False
    )
    if GreedyTokenizerFast:
        _, gt_tokenizer_alter = mock_other_tokenizer(
            {
                GreedyTokenizer: GreedyTokenizerFast,
                GreedyTokenizerFast: GreedyTokenizer,
            }[factory],
            add_bos_token=False,
            add_eos_token=False,
        )

        assert gt_tokenizer.vocab_size == gt_tokenizer_alter.vocab_size
    else:
        gt_tokenizer_alter = None

    for item in load_dataset():
        token_ids = gt_tokenizer.encode(item)
        assert gt_tokenizer.decode(token_ids) == item

        if gt_tokenizer_alter:
            assert gt_tokenizer_alter.encode(item) == token_ids
            assert gt_tokenizer_alter.decode(token_ids) == item


def test_exceptions(factory):
    with pytest.raises(TypeError):
        factory()

    if factory == GreedyTokenizer:
        with pytest.raises(UnicodeDecodeError):
            t = factory(vocab=["<0xff>"])
            t.convert_tokens_to_string(["<0xff>"])
    else:
        t = factory(vocab=["<0xff>"])
        assert t.convert_tokens_to_string(["<0xff>"]) == FAST_BYTE_FALLBACK

    buffer = UTF8Buffer("<unk>")
    buffer.push_byte(-1)
    buffer.push_byte(0x100)
    assert buffer.pop_chars() == "<unk>"

    with TemporaryDirectory() as dir_name:
        t = factory(vocab=[])
        paths = t.save_pretrained(dir_name, filename_prefix="t")
        assert any(
            "t-" + GreedyTokenizer.GREEDY_TOKENIZER_VOCAB_FILE_NAME in p for p in paths
        )


@pytest.mark.parametrize("add_bos_token", [True, False])
@pytest.mark.parametrize("add_eos_token", [True, False])
def test_bos_eos(factory, add_bos_token, add_eos_token):
    def get_tokenizer(**kwargs):
        def sub_bos_and_eos(token_id: int, token: str, _: bool) -> str:
            return {1: "<s>", 2: "</s>"}.get(token_id, token)

        _, tokenizer = mock_other_tokenizer(
            factory, proc_token=sub_bos_and_eos, **kwargs
        )

        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = "<s>"

        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "</s>"

        return tokenizer

    t = get_tokenizer(add_bos_token=add_bos_token, add_eos_token=add_eos_token)

    res = t.encode("xyz")
    assert (res[0] == t.bos_token_id) == add_bos_token  # pyright: ignore
    assert (res[-1] == t.eos_token_id) == add_eos_token  # pyright: ignore
    res = t.encode("uvw", "ijk")
    assert res.count(t.bos_token_id) == int(add_bos_token) * 2  # pyright: ignore
    assert res.count(t.eos_token_id) == int(add_eos_token) * 2  # pyright: ignore


def test_decode_special_tokens(factory):
    _, tokenizer = mock_other_tokenizer(
        factory, add_bos_token=False, add_eos_token=False
    )
    new_tokens = ["<wow>", "<yeah>"]
    tokenizer._add_tokens(new_tokens, special_tokens=True)

    assert all(
        i != tokenizer.unk_token_id for i in tokenizer.convert_tokens_to_ids(new_tokens)
    )

    def proc(s):
        tokens = tokenizer.tokenize(s)
        assert all(i in tokens for i in new_tokens)

        skipped = tokenizer.decode(tokenizer.encode(s), skip_special_tokens=True)
        assert all(i not in skipped for i in new_tokens)

        return tokenizer.decode(tokenizer.encode(s))

    assert proc("ab<wow>cd<yeah>ef") == "ab<wow>cd<yeah>ef"
    assert proc("\n \t<wow> <yeah>\t \n") == "\n \t<wow> <yeah>\t \n"

    with TemporaryDirectory() as dir_name:
        tokenizer.save_pretrained(dir_name)
        tokenizer = AutoTokenizer.from_pretrained(dir_name, trust_remote_code=True)

    assert all(
        i != tokenizer.unk_token_id for i in tokenizer.convert_tokens_to_ids(new_tokens)
    )

    assert proc("ab<wow>cd<yeah>ef") == "ab<wow>cd<yeah>ef"
    assert proc("\n \t<wow> <yeah>\t \n") == "\n \t<wow> <yeah>\t \n"


def test_fast_can_save_slow(factory):
    _, tokenizer = mock_other_tokenizer(factory)

    if GreedyTokenizerFast and isinstance(tokenizer, GreedyTokenizerFast):
        assert tokenizer.can_save_slow_tokenizer


@pytest.mark.parametrize("add_bos_token", [True, False])
@pytest.mark.parametrize("add_eos_token", [True, False])
def test_save_and_load_different_mode(factory, add_bos_token, add_eos_token):
    _, tokenizer_a = mock_other_tokenizer(
        factory, add_bos_token=add_bos_token, add_eos_token=add_eos_token
    )

    with TemporaryDirectory() as dir_name:
        tokenizer_a.save_pretrained(dir_name)
        use_fast = False if factory == GreedyTokenizerFast else True
        tokenizer_b = AutoTokenizer.from_pretrained(
            dir_name, use_fast=use_fast, trust_remote_code=True
        )

        assert tokenizer_b.add_bos_token == add_bos_token
        assert tokenizer_b.add_eos_token == add_eos_token

    if factory == GreedyTokenizerFast:
        assert type(tokenizer_a).__name__ != type(tokenizer_b).__name__
    else:
        assert type(tokenizer_a).__name__ == type(tokenizer_b).__name__
