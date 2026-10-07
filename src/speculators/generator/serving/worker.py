"""One vLLM instance per benchmark arm, with explicit request-end RPC cleanup."""

import json
import shlex
import sys
import time
from pathlib import Path

import torch

from speculators.bank.artifacts import write_json
from speculators.bank.progress import report
from speculators.generator.serving.benchmark import (
    metric_delta,
    scalar_metrics,
    summarize,
)
from speculators.generator.serving.bundle import VLLM_COMMIT, VLLM_VERSION
from speculators.generator.serving.patch import PATCH_DIR
from speculators.provenance import atomic_write, package_versions


def reset_memory(worker):
    """Public collective_rpc callable, executed inside each GPU worker."""
    del worker
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def adapter_stats(worker, request_id):
    drafter = getattr(worker.model_runner, "drafter", None)
    controller = getattr(drafter, "draft_adapter", None)
    record = controller.finish(request_id) if controller is not None else {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0
    return {
        "generator_invocations": 0,
        "generator_seconds": 0.0,
        "merge_seconds": 0.0,
        "restore_seconds": 0.0,
        **record,
        "peak_gpu_allocated_bytes": peak,
    }


def run_worker(job, *, llm_type=None, sampling_type=None):
    if llm_type is None:
        import vllm  # noqa: PLC0415

        if vllm.__version__ != VLLM_VERSION:
            raise ValueError(f"Benchmark requires vLLM {VLLM_VERSION}")
        llm_type, sampling_type = vllm.LLM, vllm.SamplingParams
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "engine_args.json", job["llm_args"])
    atomic_write(
        output / "vllm_command.txt",
        "\n".join(
            [
                f"# vLLM {VLLM_VERSION}; source commit {VLLM_COMMIT}",
                *package_versions(),
                shlex.join(sys.argv),
                "# LLM constructor arguments:",
                json.dumps(job["llm_args"], sort_keys=True),
            ]
        )
        + "\n",
    )
    atomic_write(
        output / "vllm.patch", (PATCH_DIR / "vllm-0.31.0-generator.patch").read_text()
    )
    llm = llm_type(**job["llm_args"])
    prompts = job["prompts"]

    def generate(prompt):
        params = sampling_type(
            temperature=0.0, max_tokens=prompt["max_tokens"], seed=job["seed"]
        )
        return llm.generate(
            {"prompt_token_ids": prompt["prompt_token_ids"]}, params, use_tqdm=False
        )[0]

    for i in range(job["warmups"]):
        reply = generate(prompts[i % len(prompts)])
        llm.collective_rpc(adapter_stats, args=(reply.request_id,))
    requests = []
    for repetition in range(job["repetitions"]):
        for i, prompt in enumerate(prompts):
            before = scalar_metrics(llm.get_metrics())
            llm.collective_rpc(reset_memory)
            started = time.perf_counter()
            reply = generate(prompt)
            statistics = llm.collective_rpc(adapter_stats, args=(reply.request_id,))
            seconds = time.perf_counter() - started
            if len(statistics) != 1:
                raise ValueError("Benchmark requires one GPU worker")
            stats = statistics[0]
            expected = int(job["arm"].endswith("generator"))
            if stats["generator_invocations"] != expected:
                raise RuntimeError(
                    "Generator did not execute exactly once for this request"
                )
            after = scalar_metrics(llm.get_metrics())
            requests.append(
                {
                    "row": prompt["row"],
                    "repetition": repetition,
                    "prompt_tokens": len(prompt["prompt_token_ids"]),
                    "max_tokens": prompt["max_tokens"],
                    "request_seconds": seconds,
                    "output_token_ids": list(reply.outputs[0].token_ids),
                    **metric_delta(before, after),
                    **stats,
                }
            )
            report(
                f"arm={job['arm']} pass={repetition + 1}/{job['repetitions']} "
                f"prompt={i + 1}/{len(prompts)}; elapsed={seconds:.2f}s"
            )
    result = {"arm": job["arm"], "requests": requests, "summary": summarize(requests)}
    write_json(output / "result.json", result)
    return result


def main():
    run_worker(json.loads(Path(sys.argv[1]).read_text()))


if __name__ == "__main__":
    main()
