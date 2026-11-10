# Run on the VM with: colab exec -s NAME -f status.py   (used by watch.sh)
# Prints "STATUS=<word> <timestamp> ALIVE=<bool>", GPU use, and the job log tail.
import os, subprocess
STATUS_FILE = "/content/drive/MyDrive/PROJECT/logs/JOB_NAME/STATUS"  # edit: THIS job's file only
PID_FILE = "/content/run.pid"
LOCAL_LOG = "/content/job.log"
st = open(STATUS_FILE).read().strip() if os.path.exists(STATUS_FILE) else "NO_STATUS -"
pid = open(PID_FILE).read().strip() if os.path.exists(PID_FILE) else ""
print(f"STATUS={st} ALIVE={bool(pid) and os.path.exists(f'/proc/{pid}')}")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
print(subprocess.run(["tail", "-n", "15", LOCAL_LOG], capture_output=True, text=True).stdout)
