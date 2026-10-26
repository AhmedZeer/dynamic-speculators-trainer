"""Build a manually commissioned report from its frozen local evidence files."""
# ruff: noqa: INP001, T201 -- standalone build script with console output

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LABELS = {
    "eagle3-base": "EAGLE3 base",
    "trained-lora": "Trained LoRA",
    "generator-24k-2000step": "Generator, 2K label",
    "generator-4000step": "Generator, 4K label",
}


def escaped(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def table(caption, label, headings, rows):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{@{}l" + "r" * (len(headings) - 1) + "@{}}",
        r"\toprule",
        " & ".join(
            r"\shortstack{Math\\reasoning}"
            if heading == "Math reasoning"
            else escaped(heading)
            for heading in headings
        )
        + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(map(escaped, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def prepare(work, metadata):
    for name in ("main.tex", "sections", "data", "assets"):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, work / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, work / name)
    for source in (ROOT / "template").glob("*.sty"):
        shutil.copy2(source, work / source.name)
    generated = work / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "metadata.tex").write_text(
        r"\newcommand{\reportversion}{"
        + escaped(metadata["version"])
        + "}\n"
        + r"\newcommand{\evidencedate}{"
        + escaped(metadata["evidence_through"])
        + "}\n"
    )
    metrics = json.loads((ROOT / "data/comparison-v2.json").read_text())
    tables = []
    for distribution, label, heading in [
        ("in_distribution", "nemotron", ["Model", "Math", "Code", "STEM", "Chat"]),
        (
            "out_of_distribution",
            "redhat",
            ["Model", "HumanEval", "Math reasoning", "Summarization"],
        ),
    ]:
        categories = metrics["categories"][distribution]
        rows = [
            [LABELS.get(model["name"], model["name"])]
            + [
                f"{model['output_tokens_per_second'][category]:.2f}"
                for category in categories
            ]
            for model in metrics["models"]
        ]
        caption = (
            (
                "In-distribution Nemotron"
                if label == "nemotron"
                else "Out-of-distribution Red Hat"
            )
            + ": recorded aggregate output tokens/s. Correctness was not checked; "
            "step counts follow supplied labels."
        )
        tables.append(table(caption, "tab:" + label, heading, rows))
    (generated / "throughput_tables.tex").write_text("\n".join(tables))
    cache = json.loads((ROOT / "data/cache-pilot.json").read_text())
    rows = [
        [row["category"], f"{row['first']:.2f}", f"{row['warm']:.2f}"]
        for row in cache["measurements_seconds"]
    ]
    (generated / "cache_table.tex").write_text(
        table(
            (
                "Validation wall seconds from the cache pilot. "
                "The first pass includes extraction and startup effects."
            ),
            "tab:cache",
            ["Category", "First pass (s)", "Warm pass (s)"],
            rows,
        )
    )
    prepare_extended_tables(generated)


def prepare_extended_tables(generated):
    """Derive tables from frozen sources without filling missing measurements."""

    def read(name):
        return json.loads((ROOT / "data" / name).read_text())

    def write(name, caption, headings, rows):
        (generated / f"{name}.tex").write_text(
            table(caption, "tab:" + name, headings, rows)
        )

    categories = ["math", "code", "stem", "chat"]
    headings = ["Model / C", "Math", "Code", "STEM", "Chat"]
    shared = read("shared-concurrency.json")
    models = {
        model["name"]: {
            row["concurrency"]: row["output_tokens_per_second"]
            for row in model["measurements"]
        }
        for model in shared["models"]
    }
    short = {
        "eagle3-base": "EAGLE3",
        "trained-lora": "Static LoRA",
        "generator-24k-2000step": "G2 label",
        "generator-4000step": "G4 label",
    }
    for group, label, cols in [
        ("in_distribution", "shared_id", headings),
        (
            "out_of_distribution",
            "shared_ood",
            ["Model / C", "HumanEval", "Math reasoning", "Summarization"],
        ),
    ]:
        cats = shared["category_groups"][group]
        rows = [
            [f"{short[name]} / {c}"] + [f"{values[cat]:.2f}" for cat in cats]
            for name, measurements in models.items()
            for c, values in measurements.items()
        ]
        write(
            label,
            "BATCH-001: fixed-cohort output tokens/s. C1 values are "
            "reused from GEN-003, not independent repeats. G2/G4 are labels.",
            cols,
            rows,
        )

    rows = []
    for c in [1, 2, 4, 8]:
        changes = []
        for base, group in [
            ("eagle3-base", "in_distribution"),
            ("eagle3-base", "out_of_distribution"),
            ("trained-lora", "in_distribution"),
            ("trained-lora", "out_of_distribution"),
        ]:
            cats = shared["category_groups"][group]
            changes.append(
                sum(
                    100
                    * (models["generator-4000step"][c][cat] / models[base][c][cat] - 1)
                    for cat in cats
                )
                / len(cats)
            )
        rows.append([c] + [f"{value:+.2f}%" for value in changes])
    write(
        "shared_gains",
        "BATCH-001: unweighted means of per-category "
        "throughput changes for G4. These are not pooled-workload speedups.",
        ["C", "ID/base", "OOD/base", "ID/LoRA", "OOD/LoRA"],
        rows,
    )

    detail = read("shared-detail.json")["measurements"]
    lookup = {(r["model"], r["category"]): r for r in detail}

    def shared_value(name, concurrency, category, metric):
        return lookup[(f"{name}-c{concurrency}", "nemotron/nemo-" + category)][metric]

    def acceptance_percent(row):
        counters = row["acceptance"]
        return 100 * counters["num_accepted_tokens"] / counters["num_draft_tokens"]

    write(
        "shared_acceptance",
        "BATCH-001: ID acceptance length at C8, "
        "including the baseline token; higher is better.",
        headings,
        [
            [short[name] + " / 8"]
            + [
                f"{shared_value(name, 8, cat, 'acceptance_length'):.3f}"
                for cat in categories
            ]
            for name in ["eagle3-base", "trained-lora", "generator-4000step"]
        ],
    )
    write(
        "shared_ttft",
        "BATCH-001: median TTFT (ms) under closed cohorts. "
        "These measurements do not belong to BATCH-002 checkpoints.",
        headings,
        [
            [short[name] + f" / {c}"]
            + [
                f"{shared_value(name, c, cat, 'ttft_median_ms'):.1f}"
                for cat in categories
            ]
            for c in [1, 8]
            for name in short
        ],
    )

    conditioning = read("conditioning-partial.json")
    write(
        "conditioning",
        "BATCH-002: supplied ID output tokens/s at C8; "
        "24K examples and 1,000 updates per arm are intended matched budgets. "
        "K8, acceptance and TTFT have not been supplied.",
        ["Arm", "Math", "Code", "STEM", "Chat"],
        [
            [
                "Static LoRA"
                if row["reported_conditioning_examples"] is None
                else f"Generator K{row['reported_conditioning_examples']}"
            ]
            + [
                f"{row['output_tokens_per_second']['nemo-' + cat]:.2f}"
                for cat in categories
            ]
            for row in conditioning["models"]
        ],
    )

    per_request = read("per-request-summary.json")["results"]
    rows, rates, latencies = [], [], []
    for c in [1, 2, 4]:
        for name, label in [
            ("eagle3-vanilla", "Native base"),
            ("generator-4000-per-request", "Per-request G4"),
        ]:
            key = name + f"-c{c}"
            values = [per_request[key]["nemotron/nemo-" + cat] for cat in categories]
            rows.append(
                [f"{label} / {c}"]
                + [
                    f"{r['performance'][0]['output_tokens_per_second']:.2f}"
                    for r in values
                ]
            )
            rates.append(
                [f"{label} / {c}"] + [f"{acceptance_percent(r):.2f}" for r in values]
            )
            latencies.append(
                [f"{label} / {c}"]
                + [f"{r['performance'][0]['ttft_median_ms']:.2f}" for r in values]
            )
        rates.append(
            [f"Old shared G4 / {c}"]
            + [
                f"{100 * (length - 1) / 3:.2f}"
                for cat in categories
                for length in [
                    shared_value("generator-4000step", c, cat, "acceptance_length")
                ]
            ]
        )
    write(
        "per_request_tps",
        "BATCH-003: aggregate ID output tokens/s, native "
        "continuous admission. Baseline source identity is intended but not "
        "independently verifiable from the supplied summary.",
        headings,
        rows,
    )
    write(
        "per_request_rates",
        "Acceptance rate (percent). Native base and "
        "per-request G4 are BATCH-003; old shared G4 is BATCH-001. Old-shared "
        "rates use (length minus 1)/3. Cross-study rows are descriptive, "
        "not a controlled scheduling or adapter-policy ablation.",
        headings,
        rates,
    )
    write(
        "per_request_ttft",
        "BATCH-003: median TTFT (ms), measured by the client; lower is better.",
        headings,
        latencies,
    )

    rank = read("rank-sweep.json")
    baseline = rank["baseline"]
    rows = [
        [
            "Base",
            f"{baseline['acceptance_percent']:.2f}%",
            "--",
            f"{baseline['tokens_per_second']:.1f}",
            "--",
        ]
    ]
    rows.extend(
        [
            [
                r["rank"],
                f"{r['acceptance_percent']:.2f}%",
                f"{r['derived']['acceptance_gain_pp']:+.2f}",
                f"{r['tokens_per_second']:.1f}",
                f"{r['derived']['throughput_gain_percent']:+.2f}%",
            ]
            for r in rank["rows"]
        ]
    )
    write(
        "rank_serving",
        "GEN-005: screenshot-transcribed serving metrics. "
        "Gains use the displayed base model; its identity and metric "
        "aggregation are unknown.",
        ["Rank", "Accept.", "Gain (pp)", "Tok/s", "Gain"],
        rows,
    )
    write(
        "rank_geometry",
        "GEN-005: reported module-averaged adapter geometry. "
        "Gap is within-category minus between-category cosine similarity.",
        ["Rank", "Dynamism", "Within cos.", "Between cos.", "Gap"],
        [
            [
                r["rank"],
                f"{r['dyn_ratio']:.4f}",
                f"{r['cos_within']:.4f}",
                f"{r['cos_between']:.4f}",
                f"{r['derived']['cosine_gap']:.4f}",
            ]
            for r in rank["rows"]
        ],
    )
    write(
        "rank_categories",
        "GEN-005: reported dynamism by category. "
        "Definitions and normalization of these metrics are unavailable.",
        ["Rank", "Chat", "Code", "Math", "Reading"],
        [
            [r["rank"]]
            + [f"{r['dyn_' + cat]:.4f}" for cat in ["chat", "code", "math", "reading"]]
            for r in rank["rows"]
        ],
    )
    write(
        "rank_auxiliary",
        "GEN-005: additional analysis labels preserved "
        r"from the supplied summary. eff\_dim and erank are distinct metrics; "
        "their definitions and classifier protocol remain unavailable.",
        ["Rank", "eff_dim", "erank", "cat_acc", "max_cdist"],
        [
            [
                r["rank"],
                f"{r['eff_dim']:.2f}",
                f"{r['erank']:.2f}",
                f"{100 * r['cat_acc']:.1f}%",
                f"{r['max_cdist']:.4f}",
            ]
            for r in rank["rows"]
        ],
    )


def git_output(*args):
    return subprocess.check_output(  # noqa: S603 -- fixed git arguments
        [shutil.which("git"), *args], cwd=REPO, text=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler", help="Path to tectonic or pdflatex; autodetected otherwise"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate TeX inputs without compiling a PDF",
    )
    args = parser.parse_args()
    metadata = json.loads((ROOT / "report.json").read_text())
    work = ROOT / ".build"
    work.mkdir(exist_ok=True)
    prepare(work, metadata)
    if args.prepare_only:
        print(f"Prepared TeX inputs in {work}")
        return
    compiler = args.compiler or shutil.which("tectonic") or shutil.which("pdflatex")
    if not compiler:
        raise SystemExit(
            "Install tectonic or pdflatex, or supply --compiler /path/to/binary"
        )
    command = (
        [compiler, "--keep-logs", "main.tex"]
        if "tectonic" in Path(compiler).name
        else [compiler, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    )
    env = {
        **os.environ,
        "TEXINPUTS": str(work) + os.pathsep + os.environ.get("TEXINPUTS", ""),
    }
    version = subprocess.check_output(  # noqa: S603 -- explicitly selected compiler
        [compiler, "--version"], text=True
    ).splitlines()[0]
    for _ in range(1 if "tectonic" in Path(compiler).name else 2):
        result = subprocess.run(  # noqa: S603 -- explicitly selected compiler
            command, cwd=work, env=env, text=True, capture_output=True, check=False
        )
        (work / "compiler.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(
                f"Report compilation failed; inspect {work / 'compiler.log'}"
            )
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    stem = metadata["output_stem"]
    shutil.copy2(work / "main.pdf", output / f"{stem}.pdf")
    sources = [
        source
        for source in ROOT.rglob("*")
        if source.is_file()
        and not set(source.relative_to(ROOT).parts)
        & {".build", "reports", "__pycache__"}
    ]
    sources += [
        REPO / name
        for name in (
            "AGENTS.md",
            "examples/eagle3-lora/generator_campaign_online.yaml",
            "examples/eagle3-lora/generator_training_recipe.yaml",
            "examples/eagle3-lora/benchmark_suite.yaml",
            "src/speculators/generator/model.py",
            "src/speculators/generator/config.py",
            "docs/experiments/results/GEN-001-direct-data-scaling.md",
            "docs/experiments/results/GEN-004-bank-warmup-adaptation.md",
            "docs/experiments/results/assets/GEN-001-scaling-with-lora-2026-10-23.png",
            "docs/experiments/results/assets/GEN-004-adaptation-2026-10-23.png",
            "docs/experiments/results/GEN-003-generator-suite-throughput.md",
            "docs/experiments/results/README.md",
            "docs/experiments/results/GEN-002-http-token-equality.md",
            "docs/experiments/eagle3_activation_ram_cache.md",
            "docs/experiments/eagle3_generator.md",
            "docs/experiments/eagle3_generator_campaign_online.md",
            "docs/experiments/eagle3_generator_full_data.md",
            "docs/experiments/eagle3_generator_http.md",
            "docs/experiments/eagle3_benchmark_suite.md",
            "docs/experiments/speculator_benchmarks_inventory.md",
        )
    ]
    sources += [
        REPO / name
        for name in (
            "docs/experiments/results/BATCH-001-shared-adapter-concurrency.md",
            "docs/experiments/eagle3_conditioning_24k_1k.md",
            "docs/experiments/results/BATCH-003-per-request-continuous.md",
            "docs/experiments/results/GEN-005-rank-dynamism-and-serving.md",
            "docs/experiments/eagle3_per_request_lora.md",
            "docs/experiments/eagle3_per_request_lora_feasibility.md",
            "docs/experiments/eagle3_runtime_profiling.md",
            "examples/eagle3-lora/benchmark_per_request_4000.yaml",
            "examples/eagle3-lora/benchmark_conditioning_24k_1k.yaml",
            "examples/eagle3-lora/profile_generator_serial.yaml",
            "examples/eagle3-lora/profile_generator_continuous.yaml",
            "src/speculators/generator/serving/controller.py",
            "src/speculators/generator/serving/multilora.py",
            "src/speculators/generator/serving/per_request.py",
            "src/speculators/generator/serving/vanilla.py",
            "src/speculators/generator/serving/http.py",
            "src/speculators/generator/serving/endpoint.py",
            "src/speculators/generator/serving/server.py",
            "src/speculators/generator/serving/ranges.py",
            "src/speculators/benchmark/profiling.py",
            "src/speculators/benchmark/trace.py",
            "src/speculators/benchmark/config.py",
            "scripts/launch_vllm.py",
            "src/speculators/generator/serving/patches/manifest.json",
        )
    ]
    sources += sorted(
        (REPO / "src/speculators/generator/serving/patches").rglob("*.patch")
    )
    sources = sorted(set(sources))
    hashes = {
        str(source.relative_to(REPO)): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sorted(sources)
    }
    with tarfile.open(output / f"{stem}-sources.tar.gz", "w:gz") as archive:
        for source in sorted(sources):
            archive.add(source, arcname=str(source.relative_to(REPO)))
    (output / f"{stem}.patch").write_text(
        git_output(
            "diff",
            "--binary",
            "HEAD",
            "--",
            "reporting",
            "AGENTS.md",
            "docs/experiments/results",
            "docs/experiments/eagle3_generator.md",
            ":(exclude)reporting/reports",
        )
    )
    manifest = {
        **metadata,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_git_revision": git_output("rev-parse", "HEAD").strip(),
        "working_tree_dirty": bool(git_output("status", "--porcelain").strip()),
        "compiler": version,
        "command": command,
        "pdf_sha256": hashlib.sha256((output / f"{stem}.pdf").read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "missing_run_provenance": (
            "GEN-003/BATCH-001/003 lack full request-level, checkpoint, hardware "
            "and launch/source provenance; BATCH-002 supplies a partial table; "
            "GEN-005 supplies screenshots without setup or metric definitions. "
            "GPU profiler results and repeated-run uncertainty are unavailable. "
            "See individual records; report source snapshots do not replace "
            "missing per-run provenance."
        ),
    }
    (output / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / f"{stem}.pdf")


if __name__ == "__main__":
    main()
