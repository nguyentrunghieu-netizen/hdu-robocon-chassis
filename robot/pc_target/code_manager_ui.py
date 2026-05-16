import os
import shutil
import datetime
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading

# Cấu hình đường dẫn gốc của dự án
# Giả sử file này nằm trong pc_target/, thư mục gốc sẽ là thư mục cha
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED_CODE_DIR = os.path.join(PROJECT_ROOT, "shared_arduino_code")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backups")

class CodeManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Project - Code Manager")
        self.root.geometry("600x450")
        
        # Tiêu đề
        tk.Label(root, text="CODE MANAGER UI", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Frame chứa các nút chức năng
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10, fill='x', padx=20)
        
        # 1. Nút Đồng bộ Shared Code
        sync_btn = tk.Button(btn_frame, text="1. Đồng bộ Shared Code (Arduino/ESP)", command=self.sync_shared_code, bg="#4CAF50", fg="white", font=("Arial", 11))
        sync_btn.pack(fill='x', pady=5)
        
        # 2. Nút Backup Code
        backup_btn = tk.Button(btn_frame, text="2. Backup Project Theo Phiên Bản", command=self.backup_code, bg="#2196F3", fg="white", font=("Arial", 11))
        backup_btn.pack(fill='x', pady=5)
        
        # 3. Nút Build / Upload Code
        build_frame = tk.LabelFrame(root, text="3. Build / Nạp Code (Yêu cầu PlatformIO / Arduino-CLI)", padx=10, pady=10)
        build_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(build_frame, text="Chọn Thư Mục Dự Án:").grid(row=0, column=0, sticky='w')
        self.target_dir_var = tk.StringVar()
        self.dir_entry = tk.Entry(build_frame, textvariable=self.target_dir_var, width=40)
        self.dir_entry.grid(row=0, column=1, padx=5)
        
        browse_btn = tk.Button(build_frame, text="Duyệt...", command=self.browse_target_dir)
        browse_btn.grid(row=0, column=2)
        
        tk.Label(build_frame, text="Công cụ:").grid(row=1, column=0, sticky='w', pady=5)
        self.tool_var = tk.StringVar(value="PlatformIO")
        tool_combo = ttk.Combobox(build_frame, textvariable=self.tool_var, values=["PlatformIO (pio run -t upload)", "Arduino-CLI"], state="readonly")
        tool_combo.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        action_btn = tk.Button(build_frame, text="Build & Upload", command=self.run_build_upload, bg="#FF9800", fg="white")
        action_btn.grid(row=1, column=2, pady=5)
        
        # 4. OTA Update cho Pi
        ota_btn = tk.Button(root, text="4. Mở OTA Updater (Nạp cho Pi)", command=self.open_ota_updater, bg="#9C27B0", fg="white", font=("Arial", 11))
        ota_btn.pack(fill='x', padx=20, pady=5)
        
        # Console Log
        tk.Label(root, text="Console Log:").pack(anchor='w', padx=20)
        self.console = tk.Text(root, height=8, state='disabled', bg="#f4f4f4")
        self.console.pack(fill='both', padx=20, pady=5, expand=True)
        
    def log(self, message):
        self.console.config(state='normal')
        self.console.insert('end', message + "\n")
        self.console.see('end')
        self.console.config(state='disabled')

    def sync_shared_code(self):
        """Tìm các thư mục liên quan đến arduino/esp32/mega và copy shared_arduino_code vào đó"""
        if not os.path.exists(SHARED_CODE_DIR):
            self.log(f"[-] Lỗi: Không tìm thấy thư mục {SHARED_CODE_DIR}")
            return
            
        self.log(f"[*] Đang tìm các thư mục Arduino/Mega/ESP32 để đồng bộ...")
        sync_count = 0
        
        # Duyệt qua các thư mục cấp 1 và cấp 2 để tìm kiếm từ khóa
        keywords = ['arduino', 'mega', 'esp32', 'nucleo'] 
        
        for root_dir, dirs, files in os.walk(PROJECT_ROOT):
            # Bỏ qua thư mục shared gốc và thư mục ảo
            if "shared_arduino_code" in root_dir or ".venv" in root_dir or "pc_target" in root_dir:
                continue
                
            dir_name = os.path.basename(root_dir).lower()
            if any(kw in dir_name for kw in keywords):
                # Phát hiện thư mục đích, tiến hành copy
                dest_path = os.path.join(root_dir, "lib", "shared_arduino_code")
                # Nếu không dùng thư mục lib, có thể copy thẳng vào: dest_path = os.path.join(root_dir, "shared_arduino_code")
                
                try:
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(SHARED_CODE_DIR, dest_path)
                    self.log(f"[+] Đã đồng bộ tới: {os.path.relpath(dest_path, PROJECT_ROOT)}")
                    sync_count += 1
                except Exception as e:
                    self.log(f"[-] Lỗi đồng bộ vào {dir_name}: {e}")
                    
        if sync_count > 0:
            messagebox.showinfo("Thành công", f"Đã đồng bộ thư viện chung tới {sync_count} thư mục thiết bị!")
        else:
            self.log("[-] Không tìm thấy thư mục đích nào phù hợp.")

    def backup_code(self):
        """Nén toàn bộ thư mục dự án (loại trừ .venv và các thư mục backup cũ)"""
        if not os.path.exists(BACKUP_DIR):
            os.makedirs(BACKUP_DIR)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"RobotProject_v_{timestamp}"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        self.log(f"[*] Đang tạo bản sao lưu: {backup_filename}.zip ...")
        
        def run_backup():
            try:
                # Nén toàn bộ thư mục gốc
                shutil.make_archive(backup_path, 'zip', PROJECT_ROOT)
                self.log(f"[+] Sao lưu thành công: {backup_path}.zip")
                messagebox.showinfo("Backup", f"Đã sao lưu dự án thành công!\nPhiên bản: {backup_filename}.zip")
            except Exception as e:
                self.log(f"[-] Lỗi backup: {e}")
                
        threading.Thread(target=run_backup).start()

    def browse_target_dir(self):
        dir_path = filedialog.askdirectory(initialdir=PROJECT_ROOT, title="Chọn thư mục chứa Code Vi điều khiển")
        if dir_path:
            self.target_dir_var.set(dir_path)

    def run_build_upload(self):
        target_dir = self.target_dir_var.get()
        tool = self.tool_var.get()
        
        if not target_dir or not os.path.exists(target_dir):
            messagebox.showerror("Lỗi", "Vui lòng chọn thư mục chứa code vi điều khiển hợp lệ!")
            return
            
        self.log(f"[*] Đang khởi chạy {tool} trong thư mục: {os.path.basename(target_dir)}")
        
        def execute_cmd():
            cmd = ""
            if "PlatformIO" in tool:
                cmd = "pio run -t upload"
            else:
                cmd = "arduino-cli compile --upload" # Yêu cầu có FQBN, đây là ví dụ
                
            try:
                # Chạy lệnh trong thư mục đích
                process = subprocess.Popen(cmd, cwd=target_dir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    self.log(line.strip())
                process.wait()
                if process.returncode == 0:
                    self.log("[+] Nạp code thành công!")
                else:
                    self.log("[-] Nạp code thất bại! Vui lòng kiểm tra lại cấu hình công cụ.")
            except Exception as e:
                self.log(f"[-] Lỗi thực thi: {e}")
                
        threading.Thread(target=execute_cmd).start()

    def open_ota_updater(self):
        ota_script = os.path.join(PROJECT_ROOT, "pc_target", "code_updater_source.py")
        if os.path.exists(ota_script):
            self.log("[*] Đang mở công cụ OTA Updater trong cửa sổ Terminal mới...")
            # Mở cmd và chạy python script
            subprocess.Popen(["cmd.exe", "/c", "start", "python", ota_script])
        else:
            self.log("[-] Không tìm thấy file code_updater_source.py")

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeManagerApp(root)
    root.mainloop()
