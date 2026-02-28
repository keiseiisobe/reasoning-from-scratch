from reasoning_from_scratch.qwen3 import download_qwen3_small, Qwen3Model, QWEN_CONFIG_06_B, Qwen3Tokenizer, KVCache
from pathlib import Path
import torch
import time

# utils

@torch.inference_mode()
def generate_text_basic(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    input_length = token_ids.shape[1]
    model.eval()
 
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break
 
        token_ids = torch.cat(
            [token_ids, next_token], dim=1)
    return token_ids[:, input_length:]

@torch.inference_mode()
def generate_text_basic_with_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    input_length = token_ids.shape[1]
    model.eval()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]
 
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break
 
        token_ids = torch.cat(
            [token_ids, next_token], dim=1)
        out = model(next_token, cache=cache)[:, -1]
    return token_ids[:, input_length:]

def generate_stats(output_tokens_numel, begin_time, end_time, is_gpu=True, is_cache=True):
    total_time = end_time - begin_time
    device = "GPU" if is_gpu else "CPU"
    cache = " with KV Cache" if is_cache else ""
    print(f"Time on {device}{cache}: {total_time:.2f} sec")
    print(f"{int(output_tokens_numel / total_time)} tokens / sec")


# initialize tokenizer and model

qwen3_dir = "/Users/keisei/reasoning-from-scratch/qwen3/"
download_qwen3_small(kind="base", tokenizer_only=False, out_dir=qwen3_dir)

tokenizer_path = Path(qwen3_dir) / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

model_path = Path(qwen3_dir) / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))

# define shared constants

prompt = "Explain large language models in a single sentence."
max_new_tokens = 100
 
# run model and generate stats on CPU

device = torch.device("cpu")
model.to(device)
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
    ).unsqueeze(0)

begin_time = time.time()
output_token_ids_tensor = generate_text_basic(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
)
end_time = time.time()
generate_stats(output_token_ids_tensor.numel(), begin_time, end_time, False, False)

# run model and generate stats on CPU with cache

begin_time = time.time()
output_token_ids_tensor = generate_text_basic_with_cache(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
)
end_time = time.time()
generate_stats(output_token_ids_tensor.numel(), begin_time, end_time, False, True)

# run model and generate stats on GPU (Apple Silicon GPU)

device = torch.device("mps")
model.to(device)
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
    ).unsqueeze(0)

begin_time = time.time()
output_token_ids_tensor = generate_text_basic(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
)
end_time = time.time()
generate_stats(output_token_ids_tensor.numel(), begin_time, end_time, True, False)

# run model and generate stats on GPU (Apple Silicon GPU) with cache

begin_time = time.time()
output_token_ids_tensor = generate_text_basic_with_cache(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
)
end_time = time.time()
generate_stats(output_token_ids_tensor.numel(), begin_time, end_time, True, True)

# print outputs
output_text = tokenizer.decode(
    output_token_ids_tensor.squeeze(0).tolist()
)
print(f'Generated Text: {output_text}')

