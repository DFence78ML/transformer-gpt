import dataset
import gpt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

text = open("data/input.txt", encoding="utf8").read()
if __name__ == "__main__":
    seq_len=128
    traindataset = dataset.GPTDataset(
        text,
        dataset.tokenizer,
        seq_len
    )

    loader = DataLoader(
        traindataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    torch.set_num_threads(torch.get_num_threads())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    model = gpt.build_transformer(
        vocab_size=30000,
        seq_len=seq_len
    ).to(device)
    checkpoint = torch.load(
    "latest.pt",
    map_location="cpu"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )
    optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
    )      

    model.load_state_dict(checkpoint["model_state_dict"]) 

    criterion = nn.CrossEntropyLoss()

    start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch,10):
        model.train()
        print(checkpoint["epoch"])
        print(epoch)
        mask = gpt.create_causal_mask(
        seq_len
                ).to(device)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(x, mask)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(loss.item())
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"/kaggle/working/checkpoint_{epoch}.pt")
        torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f"/kaggle/working/latest.pt")
