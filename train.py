import dataset
import gpt
import torch
import torch.nn as nn
import sys
import math
import time
import config
import os
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity
import itertools
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

SEQ_LEN = config.modelconfig["seq_len"]

BATCH_SIZE = config.batch_size
GRAD_ACCUM_STEPS = config.grad_accum

MAX_EPOCHS = config.epochs
STRIDE = 128

BASE_LR = 1.5e-3

WEIGHT_DECAY = config.weight_decay
GRAD_CLIP = config.grad_clip

PATIENCE = 3

tokens_seen = 0
epoch_tokens = 0

NUM_WORKERS = 0

PROFILE_EPOCH = 0
PROFILING_BOOL = False

def debug_checkpoint(message):
    print(
        f"[Rank {rank}] {time.strftime('%H:%M:%S')} {message}",
        flush=True
    )

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
def setup_DDP():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    torch.distributed.init_process_group(
        backend="nccl"
    )

    return rank, local_rank, world_size, device

def validate(
    model,
    val_loader,
    criterion,
    device,
    mask,
    max_batches=100
):
    total_tokens = 0
    model.eval()

    total_loss = torch.tensor(0.0, dtype=torch.float64).to(device)

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
                device_type=device.type,
                enabled=(device.type == "cuda")
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

            loss = loss * y.numel()
            logging_loss = loss

            total_tokens += y.numel()

            total_loss += logging_loss
    
    total_tokens = torch.tensor(total_tokens, dtype=torch.float64).to(device)
    stats = torch.stack([total_loss, total_tokens])

    dist.all_reduce(
                stats,
                op=dist.ReduceOp.SUM
            )

    global_loss = (
                stats[0] /
                stats[1]
            )
    model.train()

    return global_loss.item()

def trace_handler(p):
        sort_by_keyword = "self_cuda_time_total"
        output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
        print(output)
        p.export_chrome_trace(f"/kaggle/working/rank_{rank}_trace_{str(p.step_num)}.json")

def training_loop(profiling_bool, train_loader, model, optimizer, scaler, criterion, device, mask, train_sampler, p, epoch_tokens, tokens_seen, total_loss, global_step, start_time, total_optimizer_steps, world_size, rank, use_amp, next_batch_idx, epoch, best_val_loss, patience_counter, raw_model, total_batches_per_epoch):
            tokens_per_sec = 0.0
            for batch_idx, (x, y) in enumerate(
                train_loader,
                next_batch_idx
            ):
                if batch_idx == 0 and rank == 0:
                    debug_checkpoint("First batch received")
                is_update_step = (
                    (batch_idx + 1)
                    % GRAD_ACCUM_STEPS == 0
                )

                is_last_batch = (
                    batch_idx + 1
                    == total_batches_per_epoch
                )

                group_start = (
                    batch_idx // GRAD_ACCUM_STEPS
                ) * GRAD_ACCUM_STEPS

                if batch_idx == group_start:
                    group_end = min(
                        group_start + GRAD_ACCUM_STEPS,
                        total_batches_per_epoch
                    )

                    group_tokens = sum(
                        min(
                            BATCH_SIZE,
                            len(train_sampler) - i * BATCH_SIZE
                        ) * SEQ_LEN
                        for i in range(group_start, group_end)
                    )
                should_sync = is_last_batch or is_update_step

                x = x.to(
                    device,
                    non_blocking=True
                )

                y = y.to(
                    device,
                    non_blocking=True
                )
                if batch_idx == 0 and rank == 0:
                    debug_checkpoint("Batch gone to GPU")

                batch_tokens = (x.size(0) * x.size(1))
                epoch_tokens += batch_tokens
                tokens_seen += batch_tokens * world_size
                if should_sync:
                    with torch.amp.autocast(
                        device_type="cuda",
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

                    loss = loss *y.numel()
                    logging_loss = loss.detach()
                    
                    loss = (
                        loss /
                        group_tokens
                    )


                    scaler.scale(
                        loss
                    ).backward()
                else:
                    with model.no_sync():
                        with torch.amp.autocast(
                            device_type="cuda",
                            enabled=use_amp
                        ):

                            logits = model(
                                x,
                                mask
                            )

                            if batch_idx == 0 and rank == 0:
                                debug_checkpoint("Forward pass")

                            loss = criterion(
                                logits.reshape(
                                    -1,
                                    logits.size(-1)
                                ),
                                y.reshape(-1)
                            )
                        loss = loss * y.numel()
                        logging_loss = loss.detach()

                        loss = (
                            loss /
                            group_tokens
                        )


                        scaler.scale(
                            loss
                        ).backward()

                total_loss += logging_loss

                if batch_idx == 0 and rank == 0:
                    debug_checkpoint("Back pass done")


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

                    if batch_idx == 0 and rank == 0:
                        debug_checkpoint("First optimizer step")

                    optimizer.zero_grad(
                        set_to_none=True
                    )
                    global_step += 1
                    elapsed = time.time() - start_time
                    if global_step % 100 == 0:
                        torch.cuda.synchronize()
                        elapsed = time.time() - start_time
                    tokens_per_sec = int(epoch_tokens / elapsed)
                    if global_step % 100 == 0:
                        print(
                            f"Step {global_step:,} | "
                            f"LR {lr:.2e}"
                            f"Time: {elapsed}"
                            f"Tokens per sec {tokens_per_sec}"
                        )
                    next_batch_idx = batch_idx + 1
                    if global_step % 100 == 0:
                        gathered_cpu_rng, gathered_cuda_rng = cpu_cuda_rng_gather(rank)
                        if rank == 0:
                            mid_train_checkpoint(epoch, global_step, raw_model, optimizer, scaler, best_val_loss, patience_counter, tokens_seen, total_optimizer_steps, gathered_cpu_rng, gathered_cuda_rng, next_batch_idx)
                if profiling_bool:
                    p.step()
            return total_loss, epoch_tokens, tokens_seen, global_step, tokens_per_sec

def mid_train_checkpoint(epoch, global_step, raw_model, optimizer, scaler, best_val_loss, patience_counter, tokens_seen, total_optimizer_steps, gathered_cpu_rng, gathered_cuda_rng, next_batch_idx):
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

                "tokens_seen": 
                tokens_seen,

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
                WEIGHT_DECAY,

                "total_optimizer_steps":
                total_optimizer_steps,

                "config": {
                    "seq_len":
                    SEQ_LEN,

                    "d_ff":
                    config.modelconfig["d_ff"],

                    "d_model":
                    config.modelconfig["d_model"],

                    "heads":
                    config.modelconfig["heads"],

                    "layers":
                    config.modelconfig["layers"],

                    "vocab_size":
                    config.modelconfig["vocab_size"],

                    "dropout":
                    config.modelconfig["dropout"]
                },

                "cpu_rng_state":
                gathered_cpu_rng,
    
                "cuda_rng_state":
                gathered_cuda_rng,

                "next_batch_idx":
                next_batch_idx
            }


            torch.save(
                checkpoint,
                "/kaggle/working/latest.pt"
            )
def cpu_cuda_rng_gather(rank):
        local_cuda_rng = torch.cuda.get_rng_state()
        local_cpu_rng = torch.get_rng_state()

        local_cpu_rng_cuda = local_cpu_rng.to(device)

        if rank == 0:
            gathered_cuda_rng = [
                torch.empty_like(local_cuda_rng).to(device)
                for _ in range(world_size)
            ]
        else:
            gathered_cuda_rng = None

        dist.gather(
            local_cuda_rng.to(device),
            gather_list=gathered_cuda_rng,
            dst=0
        )

        if rank == 0:
            gathered_cpu_rng = [
                torch.empty_like(local_cpu_rng_cuda)
                for _ in range(world_size)
            ]
        else:
            gathered_cpu_rng = None

        dist.gather(
            local_cpu_rng_cuda,
            gather_list=gathered_cpu_rng,
            dst=0
        )

        if rank == 0:
            gathered_cuda_rng = [
                state.cpu()
                for state in gathered_cuda_rng
            ]

            gathered_cpu_rng = [
                state.cpu()
                for state in gathered_cpu_rng
            ]
        return gathered_cpu_rng, gathered_cuda_rng

class ResumableSampler(Sampler):
    def __init__(self, sampler, start_batch, batch_size):
        self.sampler = sampler
        self.start_sample = start_batch * batch_size

    def __iter__(self):
        return itertools.islice(
            iter(self.sampler),
            self.start_sample,
            None
        )

    def __len__(self):
        return max(
            0,
            len(self.sampler) - self.start_sample
        )
if __name__ == "__main__":
    traindataset = dataset.GPTDataset(
        "/kaggle/input/datasets/preetsidhu20/tokenized-files/train.bin",
        SEQ_LEN,
        STRIDE
    )

    assert traindataset != None

    valdataset = dataset.GPTDataset(
        "/kaggle/input/datasets/preetsidhu20/tokenized-files/val.bin",
        SEQ_LEN,
        STRIDE
    )

    assert valdataset != None

    vocab_size = dataset.tokenizer.get_vocab_size()
    rank, local_rank, world_size, device = setup_DDP()
    model = gpt.build_transformer(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN
    ).to(device)

    model = DDP(model,
                device_ids=[local_rank])
    
    raw_model = model.module

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

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
        optimizer_steps_per_epoch = 0
        MAX_LR = config.max_lr
        MIN_LR = config.min_lr
        next_batch_idx = 0

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
        if checkpoint["config"] != config.modelconfig:
            raise RuntimeError("Checkpoint config doesn't match config model config")


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

        total_optimizer_steps = checkpoint.get(
            "total_optimizer_steps",
            0
        )

        WARMUP_STEPS = checkpoint.get(
            "warmup_steps",
            int(total_optimizer_steps * 0.01)
        )

        MIN_LR = checkpoint.get(
            "min_lr",
            config.min_lr
        )
        MAX_LR = checkpoint.get(
            "max_lr",
            config.max_lr
                )

        next_batch_idx = checkpoint.get(
            "next_batch_idx",
            0
        )

        if next_batch_idx > 0:
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = checkpoint["epoch"] + 1

        if rank == 0:
            cuda_rng_state = checkpoint.get(
                    "cuda_rng_state",
                    None
                )
            cpu_rng_state = checkpoint.get(
                    "cpu_rng_state",
                    None
                )
        else:
            cuda_rng_state = None
            cpu_rng_state = None
        cuda_recieved = torch.empty(
                torch.cuda.get_rng_state().numel(),
                dtype = torch.uint8,
                device=device
            )
        if rank == 0:
            cuda_rng_scatter = [state.to(device) for state in cuda_rng_state]
        else:
            cuda_rng_scatter = None
        dist.scatter(
            cuda_recieved,
            scatter_list=cuda_rng_scatter,
            src=0
        )

        torch.cuda.set_rng_state(
            cuda_recieved,
            device=device
        )

        cpu_rng_receive = torch.empty(
            torch.get_rng_state().numel(),
            dtype=torch.uint8,
            device=device
        )

        if rank == 0:
            cpu_rng_scatter = [
                state.to(device)
                for state in cpu_rng_state
            ]
        else:
            cpu_rng_scatter = None

        dist.scatter(
            cpu_rng_receive,
            scatter_list=cpu_rng_scatter,
            src=0
        )

        cpu_rng_receive = cpu_rng_receive.cpu()

        torch.set_rng_state(
            cpu_rng_receive
        )
    print(
        f"Rank {rank} | "
        f"Local Rank {local_rank} | "
        f"World Size {world_size} | "
        f"Device {device}"
    )

    train_sampler = DistributedSampler(
        traindataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_sampler.set_epoch(start_epoch)

    total_batches_per_epoch = math.ceil(
        len(train_sampler) / BATCH_SIZE
    )

    resumable_sampler = ResumableSampler(
        train_sampler,
        next_batch_idx,
        BATCH_SIZE
    )

    resumable_sampler.start_sample = (
        next_batch_idx * BATCH_SIZE
    )

    train_loader = DataLoader(
        traindataset,
        batch_size=BATCH_SIZE,
        sampler=resumable_sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )
    val_sampler = DistributedSampler(
            valdataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
    )

    val_sampler.set_epoch(start_epoch)

    val_loader = DataLoader(
        valdataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    optimizer_steps_per_epoch = math.ceil(
            total_batches_per_epoch / GRAD_ACCUM_STEPS
        )

    
    if load == "no":
        total_optimizer_steps = (
                    optimizer_steps_per_epoch *
                    MAX_EPOCHS
        )
        WARMUP_STEPS = int(total_optimizer_steps * 0.01)

    print(f"Device: {device}")

    train_tokens = len(traindataset.data)
    val_tokens = len(valdataset.data)
    if rank == 0:
        print()
        print("Dataset statistics")
        print("------------------")
        print(f"Train tokens:      {train_tokens:,}")
        print(f"Validation tokens: {val_tokens:,}")
        print(f"Total tokens:      {train_tokens + val_tokens:,}")

        print(f"Train sequences:   {len(traindataset):,}")
        print(f"Val sequences:     {len(valdataset):,}")

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    if rank == 0:
        print()
        print("Model statistics")
        print("----------------")
        print(f"Vocabulary:        {vocab_size:,}")
        print(f"Parameters:         {num_params:,}")
        print(f"Trainable params:   {trainable_params:,}")



    effective_batch_size = (
        BATCH_SIZE *
        GRAD_ACCUM_STEPS *
        world_size
    )

    tokens_per_optimizer_step = (
        effective_batch_size *
        SEQ_LEN
    )

 
    if rank == 0:
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

    if rank == 0:
        print()
        print(
            f"Optimizer steps / epoch: "
            f"{optimizer_steps_per_epoch:,}"
        )

        print(
            f"Total optimizer steps:   "
            f"{total_optimizer_steps:,}"
        )

    criterion = nn.CrossEntropyLoss()

    mask = gpt.create_causal_mask(
        SEQ_LEN
    ).to(device)
    for epoch in range(
        start_epoch,
        MAX_EPOCHS
    ):
        train_sampler.set_epoch(epoch)

        model.train()
        epoch_tokens = 0
        batch_tokens = 0

        total_loss = torch.tensor(0.0, dtype=torch.float64).to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        optimizer.zero_grad(
            set_to_none=True
        )

        print()
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{MAX_EPOCHS}"
            )
        if PROFILING_BOOL and PROFILE_EPOCH == epoch and rank == 0:
            with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=trace_handler,
                ) as p:
                total_loss, epoch_tokens, tokens_seen, global_step, tokens_per_sec = training_loop(PROFILING_BOOL, train_loader, model, optimizer, scaler, criterion, device, mask, train_sampler, p, epoch_tokens, tokens_seen, total_loss, global_step, start_time, total_optimizer_steps, world_size, rank, use_amp, next_batch_idx, epoch, best_val_loss, patience_counter, raw_model, total_batches_per_epoch)
        else:
            p = None
            total_loss, epoch_tokens, tokens_seen, global_step, tokens_per_sec = training_loop(False, train_loader, model, optimizer, scaler, criterion, device, mask, train_sampler, p, epoch_tokens, tokens_seen, total_loss, global_step, start_time, total_optimizer_steps, world_size, rank, use_amp, next_batch_idx, epoch, best_val_loss, patience_counter, raw_model, total_batches_per_epoch)

        epoch_tokens = torch.tensor(epoch_tokens, dtype=torch.float64).to(device)

        stats = torch.stack([total_loss, epoch_tokens])

        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM
        )

        train_loss = (
            stats[0] /
            stats[1]
        ).item()

        elapsed = time.time() - start_time

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

        gathered_cpu_rng, gathered_cuda_rng = cpu_cuda_rng_gather(rank)
        next_batch_idx = 0

        if (rank == 0):
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
            
            mid_train_checkpoint(epoch, global_step, raw_model, optimizer, scaler, best_val_loss, patience_counter, tokens_seen, total_optimizer_steps, gathered_cpu_rng, gathered_cuda_rng, next_batch_idx)

            if is_best:
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

                    "tokens_seen": 
                    tokens_seen,

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
                    WEIGHT_DECAY,

                    "total_optimizer_steps":
                    total_optimizer_steps,

                    "config": {
                        "seq_len":
                        SEQ_LEN,

                        "d_ff":
                        config.modelconfig["d_ff"],

                        "d_model":
                        config.modelconfig["d_model"],

                        "heads":
                        config.modelconfig["heads"],

                        "layers":
                        config.modelconfig["layers"],

                        "vocab_size":
                        config.modelconfig["vocab_size"],

                        "dropout":
                        config.modelconfig["dropout"]
                    },

                    "cpu_rng_state":
                    gathered_cpu_rng,
        
                    "cuda_rng_state":
                    gathered_cuda_rng,

                    "next_batch_idx":
                    next_batch_idx
                }

                torch.save(
                    checkpoint,
                    "/kaggle/working/best.pt"
                )

        if patience_counter >= PATIENCE:
            if rank == 0:
                print()
                print(
                    "Early stopping!"
                )

                print(
                    f"Best validation loss: "
                    f"{best_val_loss:.4f}"
                )
        should_stop = torch.tensor(patience_counter >= PATIENCE, dtype = torch.bool, device = device)
        dist.broadcast(should_stop, src = 0)
        if should_stop.item():
            break