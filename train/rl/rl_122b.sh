#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

# RL recipe for the 122B model: fully-async GRPO on verl.
#
# This file holds the recipe — topology, parallelism, batch shape, async
# settings. It does not launch anything by itself; source it, then run the
# launcher that ships with the verl recipe:
#
#   cd train/verl
#   source ../rl/rl_122b.sh
#   bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
#
# Run those three lines on every node. Before the first launch, start the
# sandbox service so PythonInterpreter has an endpoint:
#
#   bash recipe/deepresearch/tools/start_local_sandbox.sh
#
# See train/README.md for the full set-up, including the search and judge
# services.

set -a

# ── Things you must provide ──────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:?path to the SFT checkpoint to start RL from}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
DATASET="${DATASET:?path to the RL prompt set (parquet)}"
RAY_DATA_HOME="${RAY_DATA_HOME:?where checkpoints and rollouts are written}"

# Tool services the rollouts call. Same variables the sampling harness reads.
SERPER_KEY_ID="${SERPER_KEY_ID:?Serper API key, for search and scholar}"
JINA_API_KEYS="${JINA_API_KEYS:?Jina Reader key(s), for visit}"
SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:?model that condenses a fetched page}"
# A sandbox for PythonInterpreter: either E2B_API_KEY or a self-hosted endpoint.
# PYTHON_NODES_CONF points at a file the sandbox service writes its endpoint to.

# ── Topology ─────────────────────────────────────────────────────────────────
ROLLOUT_NNODES="${ROLLOUT_NNODES:-5}"
TRAINER_NNODES="${TRAINER_NNODES:-2}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-8}"
TRAINER_NGPUS_PER_NODE="${TRAINER_NGPUS_PER_NODE:-8}"

# ── Rollout engines ─────────────────────────────────────────────────────────
# ROLLOUT_GPU_MEMORY_UTILIZATION must leave room for a second copy of the
# weights, which a sync holds briefly alongside the current one.
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
ROLLOUT_CONCURRENCY_PER_ENGINE="${ROLLOUT_CONCURRENCY_PER_ENGINE:-36}"
ROLLOUT_ROUTING_REPLAY="${ROLLOUT_ROUTING_REPLAY:-True}"

# ── Trainer parallelism ─────────────────────────────────────────────────────
# Constraint: (DP x TP) must be divisible by (EP x ETP).
MEGATRON_TP="${MEGATRON_TP:-2}"
MEGATRON_PP="${MEGATRON_PP:-2}"
MEGATRON_CP="${MEGATRON_CP:-1}"        # the gated-delta-net layers require CP == 1
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

# ── Batch shape and schedule ────────────────────────────────────────────────
TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-50}"
ACTOR_LR_WARMUP_STEPS="${ACTOR_LR_WARMUP_STEPS:-3}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
# Horizon the LR schedule is shaped over; defaults to twice TARGET_TRAIN_STEPS.
MEGATRON_TOTAL_TRAINING_STEPS="${MEGATRON_TOTAL_TRAINING_STEPS:-$((TARGET_TRAIN_STEPS * 2))}"

# ── Async settings ──────────────────────────────────────────────────────────
# Generation and training run concurrently. A parameter version spans
# TRIGGER_PARAMETER_SYNC_STEP training steps; STALENESS_THRESHOLD bounds how
# many versions behind a rollout may be before it is discarded.
ASYNC_PARTIAL_ROLLOUT="${ASYNC_PARTIAL_ROLLOUT:-True}"
ASYNC_TRIGGER_PARAMETER_SYNC_STEP="${ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-2}"
ASYNC_STALENESS_THRESHOLD="${ASYNC_STALENESS_THRESHOLD:-11.0}"
DEEPRESEARCH_FILTER_PROMPT_BSZ="${DEEPRESEARCH_FILTER_PROMPT_BSZ:-24}"

# ── Checkpointing ───────────────────────────────────────────────────────────
SAVE_FREQ="${SAVE_FREQ:-5}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
SAVE_EVERY_STEP_CKPT="${SAVE_EVERY_STEP_CKPT:-False}"
RESUME_MODE="${RESUME_MODE:-auto}"

# ── Evaluation during training ──────────────────────────────────────────────
TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-False}"
ROLLOUT_TEST_FREQ="${ROLLOUT_TEST_FREQ:-0}"
STOP_AFTER_N_STEPS="${STOP_AFTER_N_STEPS:-$TARGET_TRAIN_STEPS}"

set +a
echo "[rl_122b] topology ${ROLLOUT_NNODES}+${TRAINER_NNODES} nodes;" \
     "rollout TP${ROLLOUT_TENSOR_PARALLEL_SIZE};" \
     "trainer TP${MEGATRON_TP} PP${MEGATRON_PP} EP${MEGATRON_EP};" \
     "${TARGET_TRAIN_STEPS} steps, save every ${SAVE_FREQ} versions"
