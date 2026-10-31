"""Prepare pinned DeepSpec EAGLE3 for serving or native generator training."""

import argparse

from speculators.benchmark.deepspec import prepare, prepare_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output/deepseek-eagle3-ttt7-vllm")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--verifier", default="models/qwen3-8b")
    parser.add_argument(
        "--training-output-root", default="output/deepseek-eagle3-ttt7-training"
    )
    args = parser.parse_args()
    path = prepare(args.output_root)
    if args.training:
        path = prepare_training(path, args.training_output_root, args.verifier)
    print(path)
