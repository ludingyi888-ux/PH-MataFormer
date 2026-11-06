"""训练 wrapper: 同时输出到终端和日志文件"""
import subprocess, sys, os, time
from datetime import datetime

log_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f'train_{datetime.now().strftime("%m%d_%H%M")}.log')

with open(log_path, 'w', encoding='utf-8') as log_f:
    log_f.write(f'Training started at {datetime.now()}\n')
    log_f.write(f'Log: {log_path}\n\n')
    log_f.flush()
    
    proc = subprocess.Popen(
        [sys.executable, '-u', 'train.py'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=os.path.dirname(__file__),
    )
    
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        log_f.write(line); log_f.flush()
    
    proc.wait()
    log_f.write(f'\nTraining ended at {datetime.now()}, exit code={proc.returncode}\n')
