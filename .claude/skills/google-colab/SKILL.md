---
name: google-colab
description: Run GPU experiments/benchmarks on the user's Google Colab account through the official `colab` CLI (google-colab-cli) from a cloud or local agent session, with Drive-persisted outputs, a detached job, and a zero-token auto-stop watcher. Use when asked to run something "on Colab", on an A100/T4/L4/H100 runtime, or to monitor/stop/resume a Colab run.
---

# Google Colab via the `colab` CLI

Goal: run long GPU jobs on the user's Colab account **without burning Claude tokens
or compute units on polling**, keep every output on Google Drive, and always stop
the VM when the job ends.

Helpers live next to this file in `scripts/`. Copy them to your scratchpad, set the
variables at the top, and use them; do not reinvent them.

## 0. Cost model — tell the user before starting

- `colab usage` prints balance and burn rate. Approximate rates: **A100 ≈ 6.7
  units/h, T4 ≈ 1.2 units/h, CPU runtime ≈ near zero**. An idle VM bills the same as
  a busy one, so never leave a VM waiting on the user.
- Use a CPU runtime (`colab new -s NAME` with no `--gpu`) for analysis that only
  reads Drive files. Use the GPU the user asks for; this project's runs used
  `--gpu A100` (40 GB).
- Estimate total time × rate and compare with the balance before launching.

## 1. One-time setup in the agent environment

```bash
S=<your scratchpad>
python -m venv "$S/colabcli" && "$S/colabcli/bin/pip" install google-colab-cli keyrings.alt
cp .claude/skills/google-colab/scripts/* "$S/"   # colab.sh finds $S/colabcli next to it
"$S/colab.sh" --auth oauth2 sessions   # first call starts the oauth2 sign-in: it prints a Google URL;
                            # the user signs in and sends back the code, which you paste on stdin
```

Sign-in is stored in the keyring file and survived container restarts here. (`colab auth`
is different: it authenticates the *VM* for GCP services, not the CLI.)

- Containers have no OS keyring: `scripts/colab.sh` sets
  `PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring`.
- **Cloud sessions need network access** to `colab.research.google.com` and to
  every VM host `*.prod.r.colab.dev` (unique per VM — the user must add a
  `*.colab.dev` wildcard in the environment's network settings; agents cannot add
  wildcards). A 403 from the proxy on either host means the allowlist is missing.
- Secrets (HF_TOKEN, WANDB_API_KEY, ...) go in a mode-600 `secrets.env` in the
  scratchpad, uploaded to `/root/.secrets.env` on the VM. Never commit, print, or
  put them in a repository, log, or Drive file. Remind the user to rotate keys
  that were pasted into chat.

## 2. Create a VM

```bash
for i in $(seq 1 15); do "$S/colab.sh" new -s bench --gpu A100 && break; sleep 120; done
```

"Service Unavailable" means no capacity: retry every 2 min in a background shell.
`colab sessions` / `colab status -s NAME` show what exists; a session missing from
`colab sessions` is gone (Colab can reclaim a VM, especially one left idle).

## 3. Mount Drive — needs the user, and needs them quickly

Each new VM needs one browser approval. `drivemount` prints an OAuth URL and then
waits for **Enter on stdin**, so feed stdin from a file:

```bash
: > "$S/enter.txt"
(setsid nohup bash -c "tail -f '$S/enter.txt' | '$S/colab.sh' drivemount -s bench > '$S/drive.log' 2>&1" >/dev/null 2>&1 &)
sleep 20; grep -o 'https://accounts[^ ]*' "$S/drive.log" | head -1   # send VERBATIM to the user
# after the user says they approved:
echo >> "$S/enter.txt"
until grep -qE "Mounted at|mount failed|Error" "$S/drive.log"; do sleep 5; done
```

- **Only create the VM when the user is at the screen.** A late approval (VM idle
  for many minutes) has ended with the VM reclaimed and the mount failing.
- Paste the link exactly as printed; one mangled character breaks it.
- If the helper exits or the first mount reports `mount failed`, restart the helper
  and send Enter immediately — the approval usually carries over to the same VM.
- **Verify the mount before launching anything** (`mount | grep ' /content/drive '`).
  If a job starts before the mount, it writes to a plain local `/content/drive`
  directory, which then blocks the real mount (`rm -rf /content/drive` after
  killing the job, then mount again).

## 4. Launch the job detached

`colab exec -s NAME -f file.py` runs a file in the VM's Jupyter kernel and blocks
until it returns. Long jobs must run in their own session so the kernel stays free:

```bash
"$S/colab.sh" upload -s bench run.sh /content/run.sh
"$S/colab.sh" upload -s bench "$S/secrets.env" /root/.secrets.env
"$S/colab.sh" exec -s bench -f scripts/launch.py    # edit JOB/STATUS paths first
```

`scripts/launch.py` uses `setsid nohup ... &`, writes the job PID to
`/content/run.pid`, and logs to **local VM disk** (`/content/job.log`), copying the
log to Drive at the end. Do not open log files on Drive from the kernel process —
an `exec` that opened a Drive file hung and wedged the kernel.

The job script (`run.sh`) must:
- write a one-line status file **unique to this job** on Drive,
  e.g. `.../logs/<job>/STATUS`, with values like
  `SETUP <ts>`, `RUNNING <ts>`, `DONE <ts>`, `FAILED:<reason> <ts>` (always with a
  timestamp — the watcher's parser expects `STATUS=<word> <ts>`);
- write every result and provenance artifact under a persistent Drive
  `--output-root`, so a lost VM loses nothing that finished;
- be resumable (rerun the same command → completed cells are skipped);
- on failure, tail the relevant service log into the job log before `FAILED`;
- optionally retry a resumable step a bounded number of times (e.g. 3 tries)
  before declaring `FAILED`, for known intermittent faults.

For this repository follow `AGENTS.md`: setup is the repo's documented install
commands; always pass `--provenance-dir` to `scripts/launch_vllm.py`; keep
`train_command.txt`/`eval_command.txt`/`vllm_command.txt`/checkpoint hashes/patches
with the outputs. If a fix is not pushed yet, upload it as a patch and `git apply`
it on the VM (make that step idempotent: `git apply -R --check p || git apply p`).

## 5. Wait without spending tokens

Never poll in the foreground or with chained `sleep`. Use three layers:

1. **Watcher** (`scripts/watch.sh`, detached with `setsid nohup`): every 15 min it
   runs `scripts/status.py` on the VM (status file, job-PID liveness, GPU use, log
   tail) plus `colab usage`, appends one line to `watch.log`, and **stops the VM**
   on `DONE`, `FAILED`, job process gone, or balance below a floor. Costs no tokens.
2. **Background wait** (Bash `run_in_background: true`, 2 h maximum) on
   `watch.log` containing `STOPPED` — this wakes the agent exactly once. Re-arm it
   when it times out; that timeout is not an event for the user.
3. **`send_later` check-ins** at the expected milestones (startup confirmed, ~ETA)
   to report progress: one status call, an ETA from completed-cell timestamps, and
   the units needed vs the balance.

Watcher rules learned the hard way — keep them:
- Point every watcher at **its own job's** status file and PID file. A watcher
  that read an older job's `DONE` stopped a fresh VM mid-setup.
- A failed `exec` (connection lost, timeout) is **not** a failure: stop only after
  4 consecutive unreachable checks (~1 h). One transient loss once killed a
  healthy run that was 70% done.
- Don't run extra `exec` calls against a VM while a watcher is checking it unless
  needed; concurrent calls have coincided with lost connections.
- Before chaining job B after job A on the same VM, make sure A's `DONE` cannot
  trigger the watcher (use a separate status file for B and restart the watcher
  on it), and never wait with `pgrep -f <name>` inside a `bash -c` whose own
  command line contains `<name>` — it matches itself forever.

## 6. Diagnose and recover

- `exec` hangs: the kernel is busy or wedged → `colab restart-kernel -s NAME`
  (detached jobs started with `setsid` survive) and retry.
- Job failed: read the job log tail and the newest service log under the output
  root; fix; rerun the same command on a new VM (resume keeps completed work).
- Only Drive files survive a VM; read them later with the Google Drive connector
  (small text files) or a CPU runtime with Drive mounted (bulk analysis).
- When done or failed: `colab stop -s NAME`, then confirm with `colab sessions`
  and report the balance.

## 7. Report to the user

Say what ran, what finished, where outputs and provenance are on Drive, units
spent, and anything that went wrong (including your own mistakes and what they
cost). Distinguish measurements from interpretation; note single-run limits.
