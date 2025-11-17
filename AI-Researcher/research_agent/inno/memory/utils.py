import tiktoken

from research_agent.constant import COMPLETION_MODEL

_ENCODER_CACHE = {}


def _get_encoding(model_name: str):
    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: model {model_name} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    _ENCODER_CACHE[model_name] = encoding
    return encoding


def encode_string_by_tiktoken(content: str, model_name: str = COMPLETION_MODEL):
    encoding = _get_encoding(model_name)
    tokens = encoding.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = COMPLETION_MODEL):
    encoding = _get_encoding(model_name)
    content = encoding.decode(tokens)
    return content


def chunking_by_token_size(
    content: str,
    overlap_token_size=128,
    max_token_size=1024,
    tiktoken_model: str = COMPLETION_MODEL,
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results