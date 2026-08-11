import dataset
import gpt
import torch
import torch.nn as nn
import sys

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
        batch_size=128,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
            valdataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = dataset.tokenizer.get_vocab_size()

    model = gpt.build_transformer(
        vocab_size=vocab_size,
        seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )

    load = sys.argv[1].lower() if len(sys.argv) > 1 else "no"

    if load == "no":
        best_val_loss = float("inf")
        start_epoch = 0
        patience_counter = 0
        pass
    else:
        if load == "latest":
            checkpoint = torch.load(
            "/kaggle/input/datasets/preetsidhu20/transformer-checkpoints/latest.pt",
            map_location="cpu"
            )
        else:
            checkpoint = torch.load(
                "/kaggle/input/datasets/preetsidhu20/transformer-checkpoints/best.pt",
                map_location="cpu"
            )

        optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
        )      

        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

        model.load_state_dict(checkpoint["model_state_dict"]) 

        start_epoch = checkpoint["epoch"] + 1

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        patience_counter = checkpoint.get(
            "patience_counter",
            0
        )

    criterion = nn.CrossEntropyLoss() 

    raw_model = model

    model = torch.compile(model)

    def validate(model, val_loader, criterion, device, mask, max_batches=100):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= max_batches:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x, mask)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                total_loss += loss.item()

        model.train()

        return total_loss / min(len(val_loader), max_batches)
    mask = gpt.create_causal_mask(
        seq_len
    ).to(device)

    patience = 3

    for epoch in range(start_epoch,10):
        total_loss = 0
        model.train()
        print(epoch)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
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

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience{patience_counter}:{patience}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "patience_counter": patience_counter,
            "best_val_loss": best_val_loss,
        }

        torch.save(
            checkpoint,
            "/kaggle/working/latest.pt"
        )

        torch.save(
            checkpoint,
            f"/kaggle/working/checkpoint_{epoch}.pt"
        )

        if is_best:
            torch.save(
                checkpoint,
                "/kaggle/working/best.pt"
            )

        if patience_counter >= patience:
            print(
                f"\nEarly stopping!"
            )
            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )
            break
