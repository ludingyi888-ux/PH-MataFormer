"""FY-4A 批量下载 — 断点续传 + 重试（凭据从环境变量读取）

环境变量:
  FY4A_FTP_HOST    FTP 主机（默认 ftp.nsmc.org.cn）
  FY4A_FTP_ORDERS  订单列表，格式: "user:pass:dir;user2:pass2:dir2"
  FY4A_DATA_DIR    本地保存目录（默认 D:\\curelsummer\\week3\\data）

用法示例 (PowerShell):
  $env:FY4A_FTP_ORDERS = "A2026...:pwd1:batch3;A2026...:pwd2:batch4"
  $env:FY4A_DATA_DIR   = "D:\\curelsummer\\week3\\data"
  python download_fy4a.py
"""
import os, sys, time
from ftplib import FTP

HOST = os.environ.get('FY4A_FTP_HOST', 'ftp.nsmc.org.cn')
BASE = os.environ.get('FY4A_DATA_DIR', r'D:\curelsummer\week3\data')

orders_raw = os.environ.get('FY4A_FTP_ORDERS', '').strip()
if not orders_raw:
    sys.exit('未设置 FY4A_FTP_ORDERS 环境变量，格式: "user:pass:dir;user2:pass2:dir2"')

ORDERS = []
for item in orders_raw.split(';'):
    item = item.strip()
    if not item:
        continue
    parts = item.split(':')
    if len(parts) != 3:
        sys.exit(f'订单格式错误（应为 user:pass:dir）: {item}')
    user, pwd, d = parts
    ORDERS.append({'user': user, 'pass': pwd, 'dir': d})

for od in ORDERS:
    d = os.path.join(BASE, od['dir']); os.makedirs(d, exist_ok=True)
    print(f"\n{od['dir']}: connecting...")
    ftp = FTP(HOST, timeout=60); ftp.login(od['user'], od['pass']); ftp.set_pasv(True)
    files = ftp.nlst(); ftp.quit()
    print(f"  {len(files)} files total")

    ok = skip = fail = 0
    for i, fn in enumerate(files):
        lp = os.path.join(d, fn)
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            skip += 1; continue

        for retry in range(3):
            try:
                ftp = FTP(HOST, timeout=120); ftp.login(od['user'], od['pass']); ftp.set_pasv(True)
                with open(lp, 'wb') as f:
                    ftp.retrbinary(f'RETR {fn}', f.write, blocksize=65536)
                ftp.quit(); ok += 1; break
            except Exception as e:
                if retry < 2: time.sleep(5)
                else:
                    fail += 1
                    if os.path.exists(lp): os.remove(lp)
                try: ftp.quit()
                except Exception: pass
                time.sleep(2)

        if ok > 0 and ok % 20 == 0:
            mb = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)
                     if os.path.isfile(os.path.join(d, x))) / 1e6
            print(f"  [{i+1}/{len(files)}] ok={ok} skip={skip} fail={fail} {mb:.0f}MB")
        time.sleep(0.3)

    mb = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)
             if os.path.isfile(os.path.join(d, x))) / 1e6
    print(f"  DONE: ok={ok} skip={skip} fail={fail} {mb:.0f}MB")

print("\nAll complete!")
