# Run on the VM with: colab exec -s NAME -f launch.py
# Starts JOB detached (survives exec return and kernel restarts), logs to local disk,
# copies the log to Drive when the job ends, and records the PID for status.py.
import subprocess
JOB = "bash /content/run.sh"                                   # edit
LOCAL_LOG = "/content/job.log"
DRIVE_LOG_DIR = "/content/drive/MyDrive/PROJECT/logs/JOB_NAME"  # edit; same dir as the STATUS file
PID_FILE = "/content/run.pid"
assert subprocess.run("mount | grep -q ' /content/drive '", shell=True).returncode == 0, "Drive is not mounted"
sh = f"{JOB} > {LOCAL_LOG} 2>&1; mkdir -p {DRIVE_LOG_DIR}; cp {LOCAL_LOG} {DRIVE_LOG_DIR}/"
subprocess.run(f"setsid nohup bash -c '{sh}' >/dev/null 2>&1 </dev/null & echo $! > {PID_FILE}", shell=True)
print("launched", open(PID_FILE).read().strip())
