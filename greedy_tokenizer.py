# /// pyproject
# [run]
# requires-python = ">=3.8"
# dependencies = ["general-sam>=0.5.3", "transformers"]
# ///

# Repository: https://github.com/ModelTC/greedy-tokenizer

# Copyright 2023 Chielo Newctle <ChieloNewctle@gmail.com>
# Copyright 2023 ModelTC Team
#
# Licensed under either of
# - Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# - MIT license: https://opensource.org/licenses/MIT
# at your option.

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, cast

from general_sam import GeneralSAM, build_trie_from_bytes
from general_sam import GreedyTokenizer as GreedyTokenizerBase
from transformers import (
    AddedToken,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

BYTE_REPR_RE = re.compile(r"<0[xX][0-9a-fA-F]{2}>")


class UTF8Buffer(object):
    def __init__(self, fallback_repr: Optional[str] = None) -> None:
        self.char_buffer: List[str] = []

        self.byte_buffer: List[int] = []
        self.capacity = 0

        self.fallback_repr = fallback_repr

        self._clean_byte_buffer()

    def pop_chars(self) -> str:
        if self.byte_buffer:
            self.push_fallback()

        assert not self.byte_buffer

        chars, self.char_buffer = self.char_buffer, []
        return "".join(chars)

    def _clean_byte_buffer(self) -> None:
        self.byte_buffer = []
        self.capacity = 0

    def push_fallback(self) -> None:
        if self.fallback_repr is None:
            raise UnicodeDecodeError(
                "utf8",
                bytes(self.byte_buffer),
                0,
                len(self.byte_buffer),
                "invalid bytes for utf8",
            )

        self._clean_byte_buffer()

        if self.char_buffer and self.char_buffer[-1] == self.fallback_repr:
            return

        self.char_buffer.append(self.fallback_repr)

    ENCODE_LENGTH = {
        (0b1110_0000, 0b1100_0000): 2,
        (0b1111_0000, 0b1110_0000): 3,
        (0b1111_1000, 0b1111_0000): 4,
    }

    @classmethod
    def get_encode_len(cls, byte: int) -> Optional[int]:
        for (mask, target), res in cls.ENCODE_LENGTH.items():
            if (byte & mask) == target:
                return res

        return None

    def push_byte(self, byte: int) -> None:
        self.byte_buffer.append(byte)

        if byte < 0 or byte > 0xFF:
            self.push_fallback()
            return

        if self.capacity == 0:
            if (byte & 0b1000_0000) == 0:
                self.char_buffer.append(bytes(self.byte_buffer).decode())
                self._clean_byte_buffer()
                return

            encode_len = self.get_encode_len(byte)
            if encode_len is None:
                return self.push_fallback()

            self.capacity = encode_len
            return

        if (byte & 0b1100_0000) != 0b1000_0000:
            self.push_fallback()
            return

        assert len(self.byte_buffer) <= self.capacity
        if len(self.byte_buffer) == self.capacity:
            self.char_buffer.append(bytes(self.byte_buffer).decode())
            self._clean_byte_buffer()


class GreedyTokenizer(PreTrainedTokenizer):
    GREEDY_TOKENIZER_VOCAB_FILE_NAME = "vocab.json"

    vocab_files_names = {
        "vocab_path": GREEDY_TOKENIZER_VOCAB_FILE_NAME,
        **PreTrainedTokenizer.vocab_files_names,
    }

    @classmethod
    def from_other_pretrained(
        cls, *args, mock_kwargs: Optional[Mapping[str, Any]] = None, **kwargs
    ):
        return cls.mock_tokenizer(
            AutoTokenizer.from_pretrained(*args, **kwargs),
            **mock_kwargs or {},
        )

    @classmethod
    def mock_tokenizer(
        cls,
        old_tokenizer: PreTrainedTokenizerBase,
        substitue_space=True,
        proc_token: Optional[Callable[[int, str, bool], str]] = None,
        **kwargs,
    ):
        old_vocab = old_tokenizer.get_vocab()
        vocab_seq = [""] * (max(old_vocab.values()) + 1)

        _proc_token = proc_token or (lambda *args: args[1])  # pyright: ignore

        if substitue_space:
            old_proc_token = _proc_token

            def _proc_token(k: int, t: str, is_special: bool) -> str:
                return old_proc_token(k, t, is_special).replace("▁", " ")

        for token, k in old_vocab.items():
            vocab_seq[k] = _proc_token(k, token, False)

        def handle_special_token(token):
            if isinstance(token, str):
                token_id = old_vocab.get(token, -1)
                return _proc_token(token_id, token, True)

            if isinstance(token, AddedToken):
                token_id = old_vocab.get(token.content, -1)
                new_token = copy.copy(token)
                new_token.content = _proc_token(token_id, token.content, True)
                return new_token

            return None

        for attr in old_tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            token = getattr(
                old_tokenizer,
                f"_{attr}",
                getattr(old_tokenizer, attr, None),
            )

            if attr == "additional_special_tokens":
                new_tokens = list(filter(bool, map(handle_special_token, token or [])))
                kwargs.setdefault(attr, new_tokens)
                continue

            new_token = handle_special_token(token)
            if new_token is not None:
                kwargs.setdefault(attr, new_token)

        for attr in ["add_bos_token", "add_eos_token"]:
            val = getattr(old_tokenizer, attr, None)
            if isinstance(val, bool):
                kwargs.setdefault(attr, val)

        return cls(vocab=vocab_seq, **kwargs)

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab: Optional[Iterable[str]] = None,
        add_bos_token=False,
        add_eos_token=False,
        **kwargs,
    ):
        self.vocab_path = vocab_path

        if vocab_path is None and vocab is None:
            raise TypeError("must specify vocab_path or vocab")

        if vocab_path is not None and vocab is None:
            with open(vocab_path) as f:
                vocab = json.load(f)

        assert vocab is not None

        self.vocab = tuple(vocab)
        self.vocab_bytes = tuple(
            self.is_byte_repr(token)
            and bytes((self.convert_byte_repr_token(token),))
            or token.encode()
            for token in self.vocab
        )

        self.token_to_id = {s: k for k, s in enumerate(self.vocab)}
        for k, s in enumerate(self.vocab):
            self.token_to_id[s] = max(self.token_to_id[s], k)

        self.token_to_id.pop("", None)

        super().__init__(
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self.add_bos_token = add_bos_token
        if self.add_bos_token:
            assert self.bos_token_id is not None

        self.add_eos_token = add_eos_token
        if self.add_eos_token:
            assert self.eos_token_id is not None

        self.token_to_id[""] = self.unk_token_id or 0

        self.trie, self.vocab_trie_node_ids = build_trie_from_bytes(self.vocab_bytes)
        self.trie_to_token_id = [self.unk_token_id or 0] * self.trie.num_of_nodes()
        for token_id, node_id in enumerate(self.vocab_trie_node_ids):
            if not self.vocab[token_id]:
                continue

            self.trie_to_token_id[node_id] = token_id

        self.trie_to_token_id[0] = self.unk_token_id or 0

        self.sam = GeneralSAM.from_trie(self.trie)

        self.base = GreedyTokenizerBase.from_sam_and_trie(self.sam, self.trie)

    @staticmethod
    def is_byte_repr(token: str) -> bool:
        return bool(BYTE_REPR_RE.fullmatch(token))

    @staticmethod
    def convert_byte_repr_token(token: str) -> int:
        return int(token[len("<0x") : -len(">")], 16)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        pieces = []
        buffer = UTF8Buffer(self.unk_token)

        def push(token):
            if not token or (
                pieces and pieces[-1] == self.unk_token and token == self.unk_token
            ):
                return

            pieces.append(token)

        for token in tokens:
            if self.is_byte_repr(token):
                buffer.push_byte(self.convert_byte_repr_token(token))
                continue

            push(buffer.pop_chars())
            push(token)

        push(buffer.pop_chars())

        return "".join(pieces)

    def get_vocab(self) -> Dict[str, int]:
        return self.token_to_id

    def _convert_token_to_id(self, token):
        return self.token_to_id[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.vocab[index]

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return len(self.vocab)

    def _tokenize(self, text, **_):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        if isinstance(text, str):
            text = text.encode()

        assert isinstance(text, bytes)

        return [
            self._convert_id_to_token(self.trie_to_token_id[k])
            for k, _ in self.base.tokenize_bytes(text)
        ]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        save_dir = Path(save_directory)

        if filename_prefix is not None:
            filename_prefix += "-"

        vocab_file_name = (
            filename_prefix or ""
        ) + self.GREEDY_TOKENIZER_VOCAB_FILE_NAME
        vocab_file_path = save_dir / vocab_file_name

        with open(vocab_file_path, "w") as f:
            json.dump(self.vocab, f)

        return (str(vocab_file_path),)

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed.
        [What are token type IDs?](../glossary#token-type-ids)

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        prefix_cnt = int(self.add_bos_token)
        suffix_cnt = int(self.add_eos_token)

        first_part = prefix_cnt + len(token_ids_0) + suffix_cnt
        second_part = prefix_cnt + len(token_ids_1 or []) + suffix_cnt

        return [0] * first_part + [1] * second_part

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence
        for sequence classification tasks by concatenating and adding special tokens.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        prefix = [self.bos_token_id] if self.add_bos_token else []
        suffix = [self.eos_token_id] if self.add_eos_token else []

        if token_ids_1 is not None:
            last = [*prefix, *token_ids_1, *suffix]
        else:
            last = []

        output = [*prefix, *token_ids_0, *suffix, *last]
        return output

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        return super()._decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=cast(bool, clean_up_tokenization_spaces),
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )


GreedyTokenizer.register_for_auto_class()
