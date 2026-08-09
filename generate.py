import torch
import torch.nn as nn
import dataset
import gpt
import train

model = gpt.build_transformer(
    vocab_size=30000,
    seq_len=128
)
checkpoint = torch.load(
    "latest.pt",
    map_location="cuda" if torch.cuda.is_available() else "cpu"
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

again=True
model.eval()
def generate(
    model,
    ids,
    max_new_tokens,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
):
    model.eval()

    for _ in range(max_new_tokens):
        input_ids = ids[-128:]

        x = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        mask = gpt.create_causal_mask(
            x.size(1)
        ).to(device)

        with torch.no_grad():
            logits = model(x, mask)

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

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        ids.append(next_token)

    return ids
while again:
    length = int(input("How long"))

    user_text = input("You: ")

    if user_text.lower() == "quit":
        again = False
        break

    prompt = "User: " + user_text + "\nAssistant:"

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

    print("Assistant:", response)