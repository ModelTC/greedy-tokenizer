# Greedy Tokenizer

[![Source file](https://img.shields.io/badge/source_file-greedy__tokenizer.py-green)](./greedy_tokenizer.py)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-informational.svg)](#license)
[![Build status](https://github.com/ModelTC/greedy-tokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/ModelTC/greedy-tokenizer/actions)

Greedily tokenize strings with the longest tokens iteratively,
compatible with `transformers.PretrainedTokenizer` and `transformers.AutoTokenizer`.

## Requirements

- [transformers](https://github.com/huggingface/transformers)
- [general-sam](https://github.com/ModelTC/general-sam)

## Installation

```sh
git clone https://github.com/ModelTC/greedy-tokenizer.git
cd greedy-tokenizer
pip install -e .
```

Or use the [source file](./greedy_tokenizer.py) directly.

## Usage

```python
from greedy_tokenizer import GreedyTokenizer
from transformers import AutoTokenizer

# Construct GreedyTokenizer with other PretrainedTokenizer
tokenizer = GreedyTokenizer.from_other_pretrained(
    "internlm/internlm-chat-7b-v1_1",
    trust_remote_code=True,
    revision="main",
)
# Or, you can use:
# old_tokenizer = AutoTokenizer.from_pretrained(...)
# tokenizer = GreedyTokenizer.mock_tokenizer(old_tokenizer)

seq = "Hello! 你好呀！🌠"
tokens = tokenizer.tokenize(seq)

print(tokens)
# ['Hello', '!', ' ', '你好', '呀', '！', '<0xF0>', '<0x9F>', '<0x8C>', '<0xA0>']

assert tokenizer.convert_tokens_to_string(tokens) == seq

# GreedyTokenizer can also be saved and loaded
tokenizer.save_pretrained("/tmp/codellama-gt")
tokenizer = AutoTokenizer.from_pretrained("/tmp/codellama-gt", trust_remote_code=True)

# No subwords required!
gt = GreedyTokenizer(vocab=[f'<0x{i:02x}>' for i in range(256)] + ['你好呀'])
print(gt.tokenize('你好你好呀'))
# ['<0xe4>', '<0xbd>', '<0xa0>', '<0xe5>', '<0xa5>', '<0xbd>', '你好呀']
```

## Tests

```sh
pip install -e ".[test]"
pytest -s
# You can set some environment variables
# DATASET=happylkx/InstructCoder COLUMN=input pytest -s
```

## License

- &copy; 2023 Chielo Newctle \<[ChieloNewctle@gmail.com](mailto:ChieloNewctle@gmail.com)\>
- &copy; 2023 ModelTC Team

This project is licensed under either of

- [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) ([`LICENSE-APACHE`](LICENSE-APACHE))
- [MIT license](https://opensource.org/licenses/MIT) ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

The [SPDX](https://spdx.dev) license identifier for this project is `MIT OR Apache-2.0`.
