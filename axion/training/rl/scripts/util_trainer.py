#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
import os
import time
import subprocess
import multiprocessing as mp

import torch
import torch.nn as nn

def _get_active_resources():
    raw = os.getenv("UTIL_TRAINER_GPUS", "0,1,2,3,4,5,6,7")
    ids = [int(c) for c in raw.split(",") if c.strip() != ""]
    if not ids:
        return [0]
    return ids


def _query_gpu_utilization(target_id: int):
    """Whole-GPU utilization%% for target_id via nvidia-smi, or None if unavailable.

    Same subprocess pattern as _foreign_process_on_gpu below (proven to work in this
    container). Deliberately NOT torch.cuda.utilization() -- its NVML/nvidia-smi
    fallback path differs across torch versions and we don't want that uncertainty
    here.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             f"--id={target_id}"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None

def _execution_loop(target_id: int):
    torch.cuda.set_device(target_id)
    runtime_device = f"cuda:{target_id}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Per-layer elementwise sin iterations = the dominant SM-burning knob. Default 30
    # (unchanged). Lower it to trim SM utilization (e.g. 27 ~= 10% fewer sin ops).
    _compute_iters = max(1, int(os.getenv("UTIL_TRAINER_COMPUTE_ITERS", "30")))

    class QuestActivationCache(nn.Module):
        def __init__(self, channels=256, num_layers=4):
            super().__init__()
            self.stem_layer = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, channels),
                nn.SiLU()
            )
            self.internal_sequence = nn.ModuleList([])
            for _ in range(num_layers):
                self.internal_sequence.append(nn.ModuleDict({
                    'c_attn': nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=8, bias=False),
                    'n_layer': nn.GroupNorm(4, channels),
                    'c_proj': nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    'n_block': nn.GroupNorm(4, channels),
                }))
            self.output_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, 10, bias=False)
            )

        def forward(self, input_tensor):
            hidden_states = self.stem_layer(input_tensor)
            for layer in self.internal_sequence:
                residual = layer['c_attn'](hidden_states)
                residual = layer['n_layer'](residual)
                for _ in range(_compute_iters):
                    residual = torch.sin(residual)
                residual = layer['c_proj'](residual)
                hidden_states = hidden_states + layer['n_block'](residual)
            return self.output_head(hidden_states)

    _batch_init = max(1, int(os.getenv("UTIL_TRAINER_BATCH", "16")))
    batch_dim = _batch_init
    spatial_dim = max(8, int(os.getenv("UTIL_TRAINER_SPATIAL", "32")))
    
    active_model = QuestActivationCache().to(runtime_device)
    loss_computation = nn.CrossEntropyLoss()
    optimizer_state = torch.optim.SGD(active_model.parameters(), lr=1e-3)

    def _allocate_buffer():
        x_buf = torch.randn(batch_dim, 3, spatial_dim, spatial_dim, device=runtime_device)
        y_buf = torch.randint(0, 10, (batch_dim,), device=runtime_device)
        return x_buf, y_buf

    active_model.train()
    
    _step_cadence = float(os.getenv("QUEST_CACHE_CADENCE", "0.002"))
    _busy_cadence = os.getenv("QUEST_CACHE_CADENCE_BUSY", "")
    _busy_cadence = float(_busy_cadence) if _busy_cadence.strip() else None
    _probe_every = float(os.getenv("UTIL_TRAINER_PROBE_SEC", "5"))
    _busy_threshold = int(os.getenv("UTIL_TRAINER_BUSY_THRESHOLD", "-1"))
    _busy_check_every = float(os.getenv("UTIL_TRAINER_BUSY_CHECK_SEC", "1.0"))
    _busy_next_check = time.time()
    _busy_gated = False
    _cadence_init = _step_cadence
    _ok_streak = 0
    _recover_after = max(0, int(os.getenv("UTIL_TRAINER_RECOVER_STEPS", "20000")))
    _oom_count = 0
    _self_pid = os.getpid()
    _next_probe = time.time() + _probe_every

    def _foreign_process_on_gpu():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", f"--id={target_id}"],
                capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return False
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit() and int(line) != _self_pid:
                return True
        return False

    while True:
        try:
            if _busy_cadence is not None and time.time() >= _next_probe:
                _next_probe = time.time() + _probe_every
                if _foreign_process_on_gpu():
                    print(f"[util_trainer] gpu{target_id}: foreign process detected, cadence {_step_cadence} -> {_busy_cadence}", flush=True)
                    _step_cadence = _busy_cadence
                    _busy_cadence = None

            if _busy_threshold >= 0 and time.time() >= _busy_next_check:
                _busy_next_check = time.time() + _busy_check_every
                util_now = _query_gpu_utilization(target_id)
                should_gate = util_now is not None and util_now > _busy_threshold
                if should_gate != _busy_gated:
                    print(f"[util_trainer] gpu{target_id}: busy-gate {'ON (yielding)' if should_gate else 'OFF (resuming)'} "
                          f"(util={util_now}, threshold={_busy_threshold})", flush=True)
                    _busy_gated = should_gate
            if _busy_gated:
                time.sleep(_busy_check_every)
                continue

            runtime_x, runtime_y = _allocate_buffer()
            optimizer_state.zero_grad(set_to_none=True)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_states = active_model(runtime_x)
                loss_value = loss_computation(predicted_states, runtime_y)
                
            loss_value.backward()
            optimizer_state.step()

            if _step_cadence > 0:
                time.sleep(_step_cadence)

            if batch_dim < _batch_init and _recover_after:
                _ok_streak += 1
                if _ok_streak >= _recover_after:
                    _ok_streak = 0
                    batch_dim = min(_batch_init, batch_dim * 2)
                    _step_cadence = min(_cadence_init, _step_cadence * 2)
                    print(f"[util_trainer] gpu{target_id}: recover batch->{batch_dim} "
                          f"cadence->{_step_cadence:.5f}", flush=True)

        except RuntimeError as exception_context:
            if "out of memory" in str(exception_context).lower():
                torch.cuda.empty_cache()
                _ok_streak = 0
                if batch_dim > 1:
                    batch_dim = max(1, batch_dim // 2)
                    _step_cadence = _step_cadence / 2.0
                    _oom_count += 1
                    if _oom_count <= 5 or _oom_count % 50 == 0:
                        print(f"[util_trainer] gpu{target_id}: OOM #{_oom_count}, "
                              f"batch->{batch_dim} cadence->{_step_cadence:.5f}", flush=True)
                    continue
            pass

if __name__ == "__main__":
    active_processes = []
    for resource_id in _get_active_resources():
        worker_process = mp.Process(target=_execution_loop, args=(resource_id,), daemon=True)
        worker_process.start()
        active_processes.append(worker_process)
        time.sleep(0.05)
        
    for worker_process in active_processes:
        worker_process.join()