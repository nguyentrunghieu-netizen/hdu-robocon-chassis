import socket
import threading
import time
import os
import subprocess
import base64
import zipfile
import io
from xmlrpc.server import SimpleXMLRPCServer

UDP_PORT = 50050
RPC_PORT = 50051

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Không cần kết nối thật, chỉ mượn route
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def udp_beacon():
    """Liên tục phát tín hiệu Broadcast UDP để PC tự động tìm thấy IP của Pi."""
    ip = get_ip()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    message = f"PI_UPDATER:{ip}:{RPC_PORT}".encode('utf-8')
    while True:
        try:
            sock.sendto(message, ('<broadcast>', UDP_PORT))
        except Exception:
            pass
        time.sleep(2)

# --- Các hàm RPC cho phép PC gọi từ xa ---

def ping():
    return "PONG"

def run_cmd(command):
    """Chạy lệnh shell và trả về kết quả."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}

def upload_zip(b64_zip_data, target_dir):
    """Nhận file zip (đã mã hóa base64), giải nén thẳng vào thư mục đích."""
    try:
        zip_data = base64.b64decode(b64_zip_data)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            z.extractall(target_dir)
        return {"status": "success", "msg": f"Đã giải nén thành công vào {target_dir}"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

def start_rpc_server():
    server = SimpleXMLRPCServer(("0.0.0.0", RPC_PORT), allow_none=True)
    server.register_function(ping, "ping")
    server.register_function(run_cmd, "run_cmd")
    server.register_function(upload_zip, "upload_zip")
    print(f"[*] RPC Server đang lắng nghe trên cổng {RPC_PORT}...")
    server.serve_forever()

if __name__ == "__main__":
    ip = get_ip()
    print(f"[*] Khởi động Code Updater Target trên IP: {ip}")
    # Chạy UDP Beacon chạy ẩn
    threading.Thread(target=udp_beacon, daemon=True).start()
    # Chạy Server chính
    start_rpc_server()
