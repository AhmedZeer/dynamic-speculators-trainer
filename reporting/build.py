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
    "generator-24k-2000step": "Generator, 2K steps",
    "generator-4000step": "Generator, 4K steps",
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
    for name in ("main.tex", "sections", "data"):
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
            "docs/experiments/results/README.md",
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
            "GEN-003 native workload, checkpoint, hardware, commands, and patches "
            "are not supplied; see the experiment record."
        ),
    }
    (output / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / f"{stem}.pdf")


if __name__ == "__main__":
    main()
