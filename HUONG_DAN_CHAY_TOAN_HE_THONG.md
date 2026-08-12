# Hướng Dẫn Chạy Toàn Hệ Thống (Terminal + Ngrok)

Tài liệu này gom lại đúng các câu lệnh terminal cần chạy để dựng cả 4 project liên quan, theo đúng thứ tự. Chi tiết từng tính năng xem `HUONG_DAN_DEMO.md` (project này) và `HUONG_DAN_DEMO.md` của `rag-legal-assistant`.

## 0. Cấu trúc thư mục bắt buộc

**4 thư mục phải nằm cùng cấp (anh em)** — ví dụ:

```
D:\hoc\project\
├── clone-voice-station\
├── clone-voice-client\      ← thư viện dùng chung, KHÔNG tự chạy được, không có app.py
├── rag-legal-assistant\
└── voice-lab-example\
```

**`clone-voice-client` không có lệnh chạy riêng và cũng không cần bạn tự `pip install` nó.** `rag-legal-assistant/requirements.txt` và `voice-lab-example/requirements.txt` đều có dòng:

```
-e ../clone-voice-client[local]
```

Dòng này tự cài `clone-voice-client` (dạng editable) ngay khi bạn chạy `pip install -r requirements.txt` ở 2 project đó — **với điều kiện thư mục `clone-voice-client` phải nằm cạnh chúng** (`../clone-voice-client` là đường dẫn tương đối). Thiếu thư mục này, `pip install` báo lỗi "path not found" ngay ở bước cài, và cả `rag-legal-assistant` lẫn `voice-lab-example` đều không chạy được. **Đây là lý do phải nộp/deploy đủ cả 4 thư mục, không chỉ 3.**

## 1. Cài đặt (mỗi project một lần)

```bash
cd clone-voice-station
pip install -r requirements.txt
```

```bash
cd rag-legal-assistant
pip install -r requirements.txt
```
*(lệnh trên tự cài luôn `clone-voice-client` qua dòng `-e ../clone-voice-client[local]`)*

```bash
cd voice-lab-example
pip install -r requirements.txt
```
*(cũng tự cài `clone-voice-client` tương tự)*

## 2. Chạy — mở 3 terminal riêng, đúng thứ tự

### Terminal 1 — clone-voice-station (chạy trước tiên; 2 app kia gọi vào nó)
```bash
cd clone-voice-station
python app.py
```
→ **http://127.0.0.1:8090**

### Terminal 2 — rag-legal-assistant
```bash
cd rag-legal-assistant
python app.py
```
→ **http://127.0.0.1:8000**

> Có thể thay Terminal 1 + 2 bằng 1 lệnh duy nhất: chạy `start_all.bat` trong `rag-legal-assistant` — tự khởi động `clone-voice-station` trước, đợi nó sẵn sàng rồi mới chạy app này.

### Terminal 3 — voice-lab-example
```bash
cd voice-lab-example
python app.py
```
→ **http://127.0.0.1:8091**

`clone-voice-client` không có Terminal riêng — nó chạy *bên trong* tiến trình của Terminal 2 và 3 (import như một thư viện Python bình thường), không phải một service độc lập.

## 3. Demo công khai qua ngrok (tuỳ chọn)

Mỗi app runnable (`clone-voice-station`, `rag-legal-assistant`, `voice-lab-example`) có sẵn `start_ngrok.py` riêng, lộ đúng port của app đó. Cần `NGROK_AUTHTOKEN` (lấy miễn phí tại https://dashboard.ngrok.com/tunnels/authtokens) — không hardcode token vào file, dùng biến môi trường.

### Bước 1 — Lộ clone-voice-station ra ngoài
```bash
cd clone-voice-station
set NGROK_AUTHTOKEN=your_token          # CMD
$env:NGROK_AUTHTOKEN = "your_token"     # PowerShell
python start_ngrok.py
```
In ra URL công khai, ví dụ `https://xxxx.ngrok-free.app`. Giữ cửa sổ này chạy — tunnel tắt khi đóng.

### Bước 2 — Trỏ rag-legal-assistant / voice-lab-example vào URL đó

**Cách mới (khuyến nghị — không cần restart app):**
- `rag-legal-assistant`: đăng nhập admin → menu **"Quản lý mô hình giọng nói"** → ô **"Kết nối máy chủ giọng nói (clone-voice-station)"** → dán URL ngrok → **Lưu**. Áp dụng ngay cho request tiếp theo.
- `voice-lab-example`: mở **http://127.0.0.1:8091/settings** (không cần đăng nhập) → dán URL ngrok vào ô **"Kết nối máy chủ giọng nói"** → **Lưu**.

**Cách cũ (biến môi trường, cần khởi động lại app):**
```bash
set VOICE_STATION_URL=https://xxxx.ngrok-free.app          # CMD
$env:VOICE_STATION_URL = "https://xxxx.ngrok-free.app"     # PowerShell
python app.py
```

### Bước 3 (tuỳ chọn) — Lộ luôn rag-legal-assistant / voice-lab-example ra ngoài
Nếu muốn người ngoài truy cập thẳng giao diện chat/demo (không chỉ gọi API), chạy `start_ngrok.py` tương ứng trong thư mục app đó — độc lập với Bước 1:
```bash
cd rag-legal-assistant   (hoặc voice-lab-example)
set NGROK_AUTHTOKEN=your_token          # CMD
$env:NGROK_AUTHTOKEN = "your_token"     # PowerShell
python start_ngrok.py
```

### Lưu ý
- Mỗi `start_ngrok.py` cần `NGROK_AUTHTOKEN` riêng (hoặc dùng chung 1 token cho nhiều tunnel nếu gói ngrok cho phép).
- Tunnel ngrok miễn phí đổi URL mỗi lần khởi động lại — cần lặp lại Bước 2 mỗi lần Bước 1 chạy lại.
- `clone-voice-client` không bao giờ cần lộ ra ngrok — nó không phải service, không lắng nghe port nào cả.
