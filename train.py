import dataset
import gpt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

text = open("data/input.txt", encoding="utf8").read()
split = int(0.8 * len(text))
train_text = text[:split]
val_text = text[split:]

if __name__ == "__main__":
    seq_len=128
    traindataset = dataset.GPTDataset(
        train_text,
        dataset.tokenizer,
        seq_len
    )

    valdataset = dataset.GPTDataset(
            val_text,
            dataset.tokenizer,
            seq_len
        )

    train_loader = DataLoader(
        traindataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
            valdataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

    torch.set_num_threads(torch.get_num_threads())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    model = gpt.build_transformer(
        vocab_size=30000,
        seq_len=seq_len
    ).to(device)

    checkpoint = torch.load(
    "/kaggle/input/datasets/preetsidhu20/transformer-checkpoints/latest.pt",
    map_location="cpu"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )
    optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
    )      

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    model.load_state_dict(checkpoint["model_state_dict"]) 

    criterion = nn.CrossEntropyLoss()

    start_epoch = checkpoint["epoch"] + 1

    model = torch.compile(model)

    def validate(model, val_loader, criterion, device, mask):
        model.eval()

        total_loss = 0

        with torch.no_grad():
            for x, y in val_loader:

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x, mask)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                total_loss += loss.item()

        model.train()

        return total_loss / len(val_loader)

    for epoch in range(start_epoch,10):
        total_loss = 0
        model.train()
        print(checkpoint["epoch"])
        print(epoch)
        mask = gpt.create_causal_mask(
        seq_len
                ).to(device)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=False)
            with torch.amp.autocast("cuda"):
                logits = model(x, mask)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            mask
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
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
