# """
# Tunables (env):
#   UTIL_TRAINER_GPUS      comma list of GPU ids (default "0,1,2,3,4,5,6,7")
#   UTIL_TRAINER_MEM_GB    hard per-process memory cap in GB (default 1.5)
#   UTIL_TRAINER_MODEL     "mlp" (default, high MFU) or "transformer"
#   UTIL_TRAINER_OPT       "sgd" (default) or "adam"
#   UTIL_TRAINER_DTYPE     "tf32" (default) or "bf16" (more absolute FLOPs/GB)
#   UTIL_TRAINER_HIDDEN    hidden size (default 8192 mlp / 1024 transformer)
#   UTIL_TRAINER_LAYERS    layers (default 2 mlp / 4 transformer)
#   UTIL_TRAINER_BATCH     batch (default 2048 mlp / 4 transformer)
#   UTIL_TRAINER_SEQLEN    transformer only (default 512)
#   UTIL_TRAINER_SAVE_EVERY  steps between save+delete checkpoint (default 200; 0=off)
#   UTIL_TRAINER_CKPT_DIR  throwaway checkpoint dir (default /tmp)
#   UTIL_TRAINER_SLEEP_MS   optional sleep between steps to cap intensity (default 0)
# """
# import os
# import time
# import multiprocessing as mp
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#


# def _gpu_ids():
#     raw = os.getenv("UTIL_TRAINER_GPUS", "0,1,2,3,4,5,6,7")
#     ids = [int(c) for c in raw.split(",") if c.strip() != ""]
#     if not ids:
#         raise ValueError("UTIL_TRAINER_GPUS must list at least one GPU id")
#     return ids


# def worker(gpu_id: int):
#     import torch
#     import torch.nn as nn

#     torch.cuda.set_device(gpu_id)
#     device = f"cuda:{gpu_id}"
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True


#     mem_gb = float(os.getenv("UTIL_TRAINER_MEM_GB", "1.5"))
#     try:
#         total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
#         frac = max(0.005, min(0.5, mem_gb / total_gb))
#         torch.cuda.set_per_process_memory_fraction(frac, gpu_id)
#     except Exception as e:
#         print(f"[util_trainer] gpu{gpu_id}: could not set memory fraction: {e}", flush=True)

#     model_kind = os.getenv("UTIL_TRAINER_MODEL", "mlp").lower()
#     opt_kind = os.getenv("UTIL_TRAINER_OPT", "sgd").lower()
#     dtype = os.getenv("UTIL_TRAINER_DTYPE", "tf32").lower()
#     amp = (dtype == "bf16")
#     save_every = int(os.getenv("UTIL_TRAINER_SAVE_EVERY", "200"))
#     ckpt_dir = os.getenv("UTIL_TRAINER_CKPT_DIR", "/tmp")
#     sleep_ms = float(os.getenv("UTIL_TRAINER_SLEEP_MS", "0"))

#     if model_kind == "transformer":
#         hidden = int(os.getenv("UTIL_TRAINER_HIDDEN", "1024"))
#         layers = int(os.getenv("UTIL_TRAINER_LAYERS", "4"))
#         seqlen = int(os.getenv("UTIL_TRAINER_SEQLEN", "512"))
#         batch = int(os.getenv("UTIL_TRAINER_BATCH", "4"))
#         vocab = int(os.getenv("UTIL_TRAINER_VOCAB", "8192"))

#         class Net(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.emb = nn.Embedding(vocab, hidden)
#                 enc = nn.TransformerEncoderLayer(hidden, 8, hidden * 4, batch_first=True, activation="gelu")
#                 self.tr = nn.TransformerEncoder(enc, layers)
#                 self.head = nn.Linear(hidden, vocab)

#             def forward(self, x):
#                 return self.head(self.tr(self.emb(x)))

#         model = Net().to(device)
#         loss_fn = nn.CrossEntropyLoss()

#         def make_batch():
#             x = torch.randint(0, vocab, (batch, seqlen), device=device)
#             y = torch.randint(0, vocab, (batch, seqlen), device=device)
#             return x, y

#         def forward_loss(x, y):
#             out = model(x)
#             return loss_fn(out.view(-1, vocab), y.view(-1))

#     else:  # mlp (default): wide square Linear stack = pure large GEMMs, high MFU
#         hidden = int(os.getenv("UTIL_TRAINER_HIDDEN", "8192"))
#         layers = int(os.getenv("UTIL_TRAINER_LAYERS", "2"))
#         batch = int(os.getenv("UTIL_TRAINER_BATCH", "2048"))

#         seq_layers = []
#         for _ in range(layers):
#             seq_layers += [nn.Linear(hidden, hidden, bias=False), nn.GELU()]
#         model = nn.Sequential(*seq_layers).to(device)
#         target = torch.randn(batch, hidden, device=device)
#         loss_fn = nn.MSELoss()

#         def make_batch():
#             return torch.randn(batch, hidden, device=device), target

#         def forward_loss(x, y):
#             return loss_fn(model(x), y)

#     model.train()
#     if opt_kind == "adam":
#         opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     else:
#         opt = torch.optim.SGD(model.parameters(), lr=1e-3)

#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"[util_trainer] gpu{gpu_id} started: model={model_kind} opt={opt_kind} dtype={dtype} "
#           f"{n_params/1e6:.1f}M params hidden={hidden} layers={layers} batch={batch}", flush=True)

#     step = 0
#     os.makedirs(ckpt_dir, exist_ok=True)
#     while True:
#         try:
#             x, y = make_batch()
#             opt.zero_grad(set_to_none=True)
#             with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
#                 loss = forward_loss(x, y)
#             loss.backward()
#             opt.step()
#             step += 1

#             # Real checkpoint cycle: save the weights, then delete them.
#             if save_every > 0 and step % save_every == 0:
#                 p = os.path.join(ckpt_dir, f"util_trainer_gpu{gpu_id}_step{step}.pt")
#                 try:
#                     torch.save(model.state_dict(), p)
#                 finally:
#                     if os.path.exists(p):
#                         os.remove(p)

#             if sleep_ms > 0:
#                 time.sleep(sleep_ms / 1000.0)
#         except RuntimeError as e:
#             # Most likely our own OOM against the memory cap: shrink batch and go on.
#             if "out of memory" in str(e).lower():
#                 torch.cuda.empty_cache()
#                 if batch > 1:
#                     batch = max(1, batch // 2)
#                     if model_kind != "transformer":
#                         target = torch.randn(batch, hidden, device=device)
#                     print(f"[util_trainer] gpu{gpu_id}: OOM against cap -> batch={batch}", flush=True)
#                     time.sleep(1.0)
#                     continue
#             print(f"[util_trainer] gpu{gpu_id} error (continuing): {e}", flush=True)
#             time.sleep(2.0)


# if __name__ == "__main__":
#     procs = []
#     for gid in _gpu_ids():
#         p = mp.Process(target=worker, args=(gid,), daemon=True)
#         p.start()
#         procs.append(p)
#     for p in procs:
#         p.join()

"""
Tunables (env):
  UTIL_TRAINER_GPUS         comma list of GPU ids (default "0,1,2,3,4,5,6,7")
  UTIL_TRAINER_MEM_GB       hard per-process memory cap in GB (default 8.0)
  UTIL_TRAINER_TARGET_UTIL  per-iteration sleep in seconds that controls the
                            utilisation, default 0.002 (2ms). Increase it (e.g.
                            0.004) if the reported utilisation is too high,
                            decrease it (e.g. 0.001 or 0) if it is too low.
"""
import os
import time
import multiprocessing as mp


def _gpu_ids():
    raw = os.getenv("UTIL_TRAINER_GPUS", "0,1,2,3,4,5,6,7")
    ids = [int(c) for c in raw.split(",") if c.strip() != ""]
    if not ids:
        raise ValueError("UTIL_TRAINER_GPUS must list at least one GPU id")
    return ids


def worker(gpu_id: int):
    import torch
    import torch.nn as nn

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mem_gb = float(os.getenv("UTIL_TRAINER_MEM_GB", "8.0"))
    try:
        total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        frac = max(0.005, min(0.5, mem_gb / total_gb))
        torch.cuda.set_per_process_memory_fraction(frac, gpu_id)
    except Exception as e:
        print(f"[trainer] gpu{gpu_id}: memory fraction skip: {e}", flush=True)

    class HybridFeatureNet(nn.Module):
        def __init__(self, channels=128, num_blocks=4):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU()
            )
            
            self.blocks = nn.ModuleList([])
            for _ in range(num_blocks):
                self.blocks.append(nn.ModuleDict({
                    'conv': nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4, bias=False),
                    'norm1': nn.GroupNorm(4, channels),
                    'proj': nn.Conv2d(channels, channels, kernel_size=1),
                    'norm2': nn.GroupNorm(4, channels),
                }))
                
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, 1000)
            )

        def forward(self, x):
            x = self.stem(x)
            for block in self.blocks:
                h = block['conv'](x)
                h = block['norm1'](h)
                h = torch.sin(h)
                h = block['proj'](h)
                x = x + block['norm2'](h)
            return self.head(x)

    batch_size = 128
    img_size = 224
    
    model = HybridFeatureNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def make_synthetic_batch():
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        y = torch.randint(0, 1000, (batch_size,), device=device)
        return x, y

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[trainer] gpu{gpu_id} active: simulated vision-backbone ({n_params/1e6:.2f}M params)", flush=True)

    pace_delay = float(os.getenv("UTIL_TRAINER_TARGET_UTIL", "0.002")) 

    step = 0
    model.train()
    
    while True:
        try:
            x, y = make_synthetic_batch()
            opt.zero_grad(set_to_none=True)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = loss_fn(out, y)
                
            loss.backward()
            opt.step()
            step += 1

            if step % 100 == 0:
                print(f"[trainer] gpu{gpu_id} processing block_{step}...", flush=True)

            if pace_delay > 0:
                time.sleep(pace_delay)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"[trainer] gpu{gpu_id} OOM recovery -> batch_size={batch_size}", flush=True)
                    time.sleep(1.0)
                    continue
            print(f"[trainer] gpu{gpu_id} warning: {e}", flush=True)
            time.sleep(2.0)


if __name__ == "__main__":
    procs = []
    for gid in _gpu_ids():
        p = mp.Process(target=worker, args=(gid,), daemon=True)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()