#!/usr/bin/env bash
# Launch dry-run hidden-state extraction in tmux, maximize GPU utilization.
#
# GPU 0: 14B (~28GB) + 0.5B (~1GB) = ~29GB
# GPU 1: 7B (~14GB) + 3B (~6GB) + 1.5B (~3GB) = ~23GB
#
# All processes within each GPU run in parallel.

set -euo pipefail

BREWING_DIR="/home/gyf/CUE/Brewing"
MODEL_DIR="/home/gyf/CUE/models/Qwen"
OUTPUT="$BREWING_DIR/brewing_output"
SESSION="dry_run"
CONDA_ACT="conda activate cue"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "gpu0-14b"

# --- GPU 0: 14B ---
tmux send-keys -t "$SESSION:gpu0-14b" "
cd $BREWING_DIR && $CONDA_ACT
echo '=== GPU 0: 14B ==='
CUDA_VISIBLE_DEVICES=0 python scripts/dry_run.py \
    --model-path $MODEL_DIR/Qwen2.5-Coder-14B \
    --model-id Qwen/Qwen2.5-Coder-14B \
    --gpu 0 --output-root $OUTPUT --batch-size 2
echo '=== GPU 0: 14B DONE ==='
" Enter

# --- GPU 0: 0.5B ---
tmux new-window -t "$SESSION" -n "gpu0-05b"
tmux send-keys -t "$SESSION:gpu0-05b" "
cd $BREWING_DIR && $CONDA_ACT
echo '=== GPU 0: 0.5B ==='
CUDA_VISIBLE_DEVICES=0 python scripts/dry_run.py \
    --model-path $MODEL_DIR/Qwen2.5-Coder-0.5B \
    --model-id Qwen/Qwen2.5-Coder-0.5B \
    --gpu 0 --output-root $OUTPUT --batch-size 16
echo '=== GPU 0: 0.5B DONE ==='
" Enter

# --- GPU 1: 7B ---
tmux new-window -t "$SESSION" -n "gpu1-7b"
tmux send-keys -t "$SESSION:gpu1-7b" "
cd $BREWING_DIR && $CONDA_ACT
echo '=== GPU 1: 7B ==='
CUDA_VISIBLE_DEVICES=1 python scripts/dry_run.py \
    --model-path $MODEL_DIR/Qwen2.5-Coder-7B \
    --model-id Qwen/Qwen2.5-Coder-7B \
    --gpu 0 --output-root $OUTPUT --batch-size 4
echo '=== GPU 1: 7B DONE ==='
" Enter

# --- GPU 1: 3B ---
tmux new-window -t "$SESSION" -n "gpu1-3b"
tmux send-keys -t "$SESSION:gpu1-3b" "
cd $BREWING_DIR && $CONDA_ACT
echo '=== GPU 1: 3B ==='
CUDA_VISIBLE_DEVICES=1 python scripts/dry_run.py \
    --model-path $MODEL_DIR/Qwen2.5-Coder-3B \
    --model-id Qwen/Qwen2.5-Coder-3B \
    --gpu 0 --output-root $OUTPUT --batch-size 4
echo '=== GPU 1: 3B DONE ==='
" Enter

# --- GPU 1: 1.5B ---
tmux new-window -t "$SESSION" -n "gpu1-15b"
tmux send-keys -t "$SESSION:gpu1-15b" "
cd $BREWING_DIR && $CONDA_ACT
echo '=== GPU 1: 1.5B ==='
CUDA_VISIBLE_DEVICES=1 python scripts/dry_run.py \
    --model-path $MODEL_DIR/Qwen2.5-Coder-1.5B \
    --model-id Qwen/Qwen2.5-Coder-1.5B \
    --gpu 0 --output-root $OUTPUT --batch-size 8
echo '=== GPU 1: 1.5B DONE ==='
" Enter

echo "tmux session '$SESSION' started — 5 windows, 5 models parallel"
echo "  GPU 0: 14B + 0.5B  |  GPU 1: 7B + 3B + 1.5B"
echo "  tmux attach -t $SESSION"
