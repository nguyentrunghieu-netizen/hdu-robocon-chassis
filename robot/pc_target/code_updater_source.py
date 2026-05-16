import socket
import xmlrpc.client
import os
import zipfile
import base64
import io
import sys

UDP_PORT = 50050

def discover_pi():
    """Lắng nghe gói tin UDP Broadcast từ Pi để lấy IP tự động."""
    print("[*] Đang tìm kiếm Raspberry Pi trên mạng LAN...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Lắng nghe trên mọi interface
    sock.bind(("", UDP_PORT))
    sock.settimeout(5.0) # Đợi tối đa 5 giây
    
    try:
        data, addr = sock.recvfrom(1024)
        msg = data.decode('utf-8')
        if msg.startswith("PI_UPDATER:"):
            _, ip, port = msg.split(":")
            print(f"[+] Đã tìm thấy Pi tại IP: {ip}, Port: {port}")
            return ip, int(port)
    except socket.timeout:
        print("[-] Hết thời gian chờ. Không tìm thấy Pi tự động.")
        return None, None
    finally:
        sock.close()

def zip_directory(dir_path):
    """Nén một thư mục vào bộ nhớ (RAM) và trả về bytes."""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Tính toán đường dẫn tương đối để giải nén đúng cấu trúc
                arcname = os.path.relpath(file_path, dir_path)
                zf.write(file_path, arcname)
    return memory_file.getvalue()

def main():
    ip, port = discover_pi()
    if not ip:
        ip = input("Nhập địa chỉ IP của Pi thủ công (VD: 192.168.1.100): ").strip()
        port = 50051

    pi_url = f"http://{ip}:{port}/"
    proxy = xmlrpc.client.ServerProxy(pi_url, allow_none=True)
    
    try:
        ping_res = proxy.ping()
        print(f"[+] Kết nối thành công! (Phản hồi: {ping_res})")
    except Exception as e:
        print(f"[-] Không thể kết nối tới Pi: {e}")
        return

    while True:
        print("\n" + "="*40)
        print("       PI REMOTE CONTROL & DEPLOY       ")
        print("="*40)
        print("1. Đẩy thư mục code lên Pi (Upload Folder)")
        print("2. Chạy lệnh Terminal trên Pi (Run Command)")
        print("3. Thoát")
        choice = input("\nChọn chức năng (1/2/3): ").strip()
        
        if choice == '1':
            source_dir = input("Nhập đường dẫn thư mục trên PC (VD: ./src): ").strip()
            target_dir = input("Nhập đường dẫn lưu trên Pi (VD: /home/pi/robot): ").strip()
            
            if not os.path.isdir(source_dir):
                print("[-] Thư mục nguồn không tồn tại trên PC!")
                continue
                
            print(f"[*] Đang nén thư mục '{source_dir}'...")
            zip_data = zip_directory(source_dir)
            # Mã hóa base64 để truyền qua RPC an toàn
            b64_data = base64.b64encode(zip_data).decode('utf-8')
            
            print(f"[*] Đang tải lên và giải nén vào '{target_dir}' trên Pi...")
            res = proxy.upload_zip(b64_data, target_dir)
            
            if res['status'] == 'success':
                print(f"[+] {res['msg']}")
            else:
                print(f"[-] Lỗi: {res['msg']}")
            
        elif choice == '2':
            cmd = input("Nhập lệnh shell (VD: ls -la /home/pi): ").strip()
            if not cmd: continue
            
            print(f"[*] Đang thực thi: {cmd}")
            res = proxy.run_cmd(cmd)
            
            print("\n--- KẾT QUẢ ---")
            if res['stdout']:
                print(res['stdout'])
            if res['stderr']:
                print("LỖI/CẢNH BÁO:\n" + res['stderr'])
            print(f"Mã thoát (Exit code): {res['returncode']}")
            
        elif choice == '3':
            print("Đã thoát.")
            break

if __name__ == "__main__":
    main()
