# Copyright 2023 Chielo Newctle <ChieloNewctle@gmail.com>
# Copyright 2023 ModelTC Team
#
# Licensed under either of
# - Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# - MIT license: https://opensource.org/licenses/MIT
# at your option.

import json
import os

from transformers import AutoTokenizer, PreTrainedTokenizer

from .common import bench_tokenizer


def get_tokenizer() -> PreTrainedTokenizer:
    kwargs = json.loads(os.getenv("OLD_TOKENIZER_KWARGS", "{}"))
    kwargs["use_fast"] = False
    name = os.getenv("OLD_TOKENIZER") or "codellama/CodeLlama-7b-Instruct-hf"
    return AutoTokenizer.from_pretrained(name, **kwargs)


if __name__ == "__main__":
    bench_tokenizer(get_tokenizer())
