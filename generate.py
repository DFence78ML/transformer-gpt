import torch
import torch.nn as nn
import dataset
import gpt
import config
from tokenizers.decoders import ByteLevel

vocab_size = dataset.tokenizer.get_vocab_size()

dataset.tokenizer.decoder = ByteLevel()

print(dataset.tokenizer.encode("[EOS]").ids)
print(dataset.tokenizer.encode("<|endoftext|>").ids)

model = gpt.build_transformer(
    vocab_size=vocab_size,
    seq_len=config.modelconfig["seq_len"]
)
checkpoint = torch.load(
    "best.pt",
    map_location="cpu"
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)

assert checkpoint["config"] == config.modelconfig

device = "cuda" if torch.cuda.is_available() else "cpu"

eos_id = dataset.tokenizer.encode("[EOS]").ids
end_text_id  = dataset.tokenizer.encode("<|endoftext|>").ids

again=True
model.to(device)
def generate(
    model,
    ids,
    max_new_tokens,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
):
    model.eval()

    cache = None
    next_token = 0
    tokens_printed = 0
    context_size = config.modelconfig["seq_len"]
    ids = list(ids)
    if len(ids) < 1:
        raise ValueError("Prompt needs one token")

    
    input_ids = ids[-context_size:]
    x = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

    mask = gpt.create_causal_mask(
            x.size(1)
        ).to(device)
    with torch.inference_mode():
        logits, cache = model(
            x,
            mask,
            use_cache = True,
            start_pos = 0
        )
        cache_len = x.size(1)
    for _ in range(max_new_tokens):
        input_ids = ids[-config.modelconfig["seq_len"]:]

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            values, indices = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            filtered = torch.full_like(
                logits,
                float("-inf")
            )

            filtered.scatter_(
                1,
                indices,
                values
            )

            logits = filtered
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1
            )

            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove = cumulative_probs > top_p

            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove_original = torch.zeros_like(remove).scatter(
                -1, sorted_indices, remove
            )

            logits = logits.masked_fill(remove_original, float("-inf"))

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        ids.append(next_token)
        tokens_printed += 1
        #print(tokens_printed)

        if [next_token] == end_text_id:
            break

        if cache_len < context_size:
            new_x = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=device
            )

            logits, cache = model(
                new_x,
                mask=None,
                past_key_values = cache,
                use_cache = True,
                start_pos = cache_len
            )
            cache_len += 1

        else:

            input_ids = ids[-context_size:]

            x = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            mask = gpt.create_causal_mask(
                    x.size(1)
                ).to(device)
            
            logits, cache = model(
                x,
                mask,
                past_key_values = None,
                use_cache = True,
                start_pos = 0
            )
            cache_len = x.size(1)
    return ids
while again:
    length = int(input("How long"))

    user_text = input("You: ")

    if user_text.lower() == "quit":
        again = False
        break

    prompt = user_text
    print()

    prompt_ids = dataset.tokenizer.encode(prompt).ids

    generated_ids = generate(
        model,
        prompt_ids.copy(),
        max_new_tokens=length,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    response_ids = generated_ids[len(prompt_ids):]

    response = dataset.tokenizer.decode(response_ids)

    print(f"Story:{prompt}{response}" )