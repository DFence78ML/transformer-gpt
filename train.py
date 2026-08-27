import dataset
import gpt
import torch
import torch.nn as nn
import sys
import math
import time

from torch.utils.data import DataLoader

SEQ_LEN = 128

BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 8

MAX_EPOCHS = 10

MAX_LR = 3e-4
MIN_LR = 3e-5 

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

PATIENCE = 3

tokens_seen = 0
epoch_tokens = 0

NUM_WORKERS = 4

def get_lr(step, total_steps):

    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if step >= total_steps:
        return MIN_LR

    progress = (
        step - WARMUP_STEPS
    ) / (
        total_steps - WARMUP_STEPS
    )

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LR + (
        MAX_LR - MIN_LR
    ) * cosine

def validate(
    model,
    val_loader,
    criterion,
    device,
    mask,
    max_batches=100
):

    model.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for i, (x, y) in enumerate(val_loader):

            if i >= max_batches:
                break

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device,
                enabled=(device == "cuda")
            ):

                logits = model(
                    x,
                    mask
                )

                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1)
                    ),
                    y.reshape(-1)
                )

            total_loss += loss.item()
            batches += 1

    model.train()

    return total_loss / batches

if __name__ == "__main__":
    traindataset = dataset.GPTDataset(
        "/kaggle/input/datasets/preetsidhu20/tokenized-files/train.bin",
        SEQ_LEN
    )

    valdataset = dataset.GPTDataset(
        "/kaggle/input/datasets/preetsidhu20/tokenized-files/val.bin",
        SEQ_LEN
    )

    train_loader = DataLoader(
        traindataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        valdataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    train_tokens = len(traindataset.data)
    val_tokens = len(valdataset.data)

    print()
    print("Dataset statistics")
    print("------------------")
    print(f"Train tokens:      {train_tokens:,}")
    print(f"Validation tokens: {val_tokens:,}")
    print(f"Total tokens:      {train_tokens + val_tokens:,}")

    print(f"Train sequences:   {len(traindataset):,}")
    print(f"Val sequences:     {len(valdataset):,}")

    vocab_size = dataset.tokenizer.get_vocab_size()

    model = gpt.build_transformer(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Model statistics")
    print("----------------")
    print(f"Vocabulary:        {vocab_size:,}")
    print(f"Parameters:         {num_params:,}")
    print(f"Trainable params:   {trainable_params:,}")



    effective_batch_size = (
        BATCH_SIZE *
        GRAD_ACCUM_STEPS
    )

    tokens_per_optimizer_step = (
        effective_batch_size *
        SEQ_LEN
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) /
        GRAD_ACCUM_STEPS
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch *
        MAX_EPOCHS
    )

    WARMUP_STEPS = int(total_optimizer_steps * 0.01)

    print()
    print("Training configuration")
    print("----------------------")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Accumulation steps:  {GRAD_ACCUM_STEPS}")
    print(f"Effective batch:     {effective_batch_size}")
    print(
        f"Tokens / update:     "
        f"{tokens_per_optimizer_step:,}"
    )
    print(f"Max LR:              {MAX_LR}")
    print(f"Min LR:              {MIN_LR}")
    print(f"Warmup steps:        {WARMUP_STEPS}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    print()
    print(
        f"Optimizer steps / epoch: "
        f"{optimizer_steps_per_epoch:,}"
    )

    print(
        f"Total optimizer steps:   "
        f"{total_optimizer_steps:,}"
    )

    use_amp = device == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    criterion = nn.CrossEntropyLoss()

    raw_model = model

    model = torch.compile(model)

    mask = gpt.create_causal_mask(
        SEQ_LEN
    ).to(device)

    load = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else "no"
    )

    if load == "no":

        best_val_loss = float("inf")
        start_epoch = 0
        patience_counter = 0
        global_step = 0

    else:

        if load == "latest":

            checkpoint = torch.load(
                "/kaggle/input/datasets/"
                "preetsidhu20/"
                "transformer-checkpoints/"
                "latest.pt",
                map_location="cpu"
            )

        else:

            checkpoint = torch.load(
                "/kaggle/input/datasets/"
                "preetsidhu20/"
                "transformer-checkpoints/"
                "best.pt",
                map_location="cpu"
            )


        raw_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scaler_state_dict" in checkpoint:

            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        for state in optimizer.state.values():

            for key, value in state.items():

                if torch.is_tensor(value):

                    state[key] = value.to(device)


        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        patience_counter = checkpoint.get(
            "patience_counter",
            0
        )

        global_step = checkpoint.get(
            "global_step",
            0
        )

        tokens_seen = checkpoint.get(
            "tokens_seen",
            0
        )

    for epoch in range(
        start_epoch,
        MAX_EPOCHS
    ):

        model.train()
        epoch_tokens = 0
        batch_tokens = 0

        total_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad(
            set_to_none=True
        )

        print()
        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS}"
        )


        for batch_idx, (x, y) in enumerate(
            train_loader
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            batch_tokens = (x.size(0) * x.size(1))
            epoch_tokens += batch_tokens
            tokens_seen += batch_tokens

            with torch.amp.autocast(
                device_type=device,
                enabled=use_amp
            ):

                logits = model(
                    x,
                    mask
                )

                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1)
                    ),
                    y.reshape(-1)
                )
                logging_loss = loss.item()
                loss = (
                    loss /
                    GRAD_ACCUM_STEPS
                )


            scaler.scale(
                loss
            ).backward()


            total_loss += logging_loss

            is_update_step = (
                (batch_idx + 1)
                % GRAD_ACCUM_STEPS == 0
            )

            is_last_batch = (
                batch_idx + 1
                == len(train_loader)
            )


            if is_update_step or is_last_batch:
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                lr = get_lr(
                    global_step,
                    total_optimizer_steps
                )
                for param_group in (
                    optimizer.param_groups
                ):

                    param_group["lr"] = lr

                scaler.step(
                    optimizer
                )

                scaler.update()


                optimizer.zero_grad(
                    set_to_none=True
                )
                global_step += 1

                if global_step % 100 == 0:

                    print(
                        f"Step {global_step:,} | "
                        f"LR {lr:.2e}"
                    )

        train_loss = (
            total_loss /
            len(train_loader)
        )
        elapsed = time.time() - start_time
        tokens_per_sec = int(epoch_tokens / elapsed)

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            mask
        )

        perplexity = math.exp(
            min(val_loss, 20)
        )


        print()
        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Perplexity: {perplexity:.2f}"
            f"Tokens seen this epoch: {epoch_tokens}"
            f"Tokens per sec: {tokens_per_sec}"
            f"Total tokens seen: {tokens_seen}"
        )

        is_best = (
            val_loss < best_val_loss
        )

        if is_best:

            best_val_loss = val_loss
            patience_counter = 0

        else:

            patience_counter += 1

            print(
                f"Patience "
                f"{patience_counter}/{PATIENCE}"
            )

        checkpoint = {

            "epoch": epoch,

            "global_step": global_step,

            "model_state_dict":
                raw_model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_val_loss":
                best_val_loss,

            "patience_counter":
                patience_counter,

            "tokens_seen": tokens_seen,

            "config": {

                "seq_len":
                    SEQ_LEN,

                "batch_size":
                    BATCH_SIZE,

                "grad_accum_steps":
                    GRAD_ACCUM_STEPS,

                "max_lr":
                    MAX_LR,

                "min_lr":
                    MIN_LR,

                "warmup_steps":
                    WARMUP_STEPS,

                "weight_decay":
                    WEIGHT_DECAY
            }
        }


        torch.save(
            checkpoint,
            "/kaggle/working/latest.pt"
        )


        if is_best:

            torch.save(
                checkpoint,
                "/kaggle/working/best.pt"
            )

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping!"
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )
            break