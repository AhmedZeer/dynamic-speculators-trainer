"""Prepare pinned DeepSpec EAGLE3 metadata for the existing vLLM 0.31.0 suite."""

import argparse

from speculators.benchmark.deepspec import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output/deepseek-eagle3-ttt7-vllm")
    args = parser.parse_args()
    print(prepare(args.output_root))
