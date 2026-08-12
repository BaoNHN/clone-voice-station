# Hướng Dẫn Demo — clone-voice-station

> Dịch vụ giọng nói độc lập: đọc văn bản thành giọng nói (TTS) + nhân bản giọng nói cá nhân (RVC qua Colab), phục vụ nhiều ứng dụng khách ("client") cùng lúc qua REST API. Tách ra từ dự án `rag-legal-assistant` để dùng lại cho các ứng dụng khác.

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu Cầu & Cài Đặt](#2-yêu-cầu--cài-đặt)
3. [Khởi Động Dịch Vụ](#3-khởi-động-dịch-vụ)
4. [Demo Dashboard Quản Trị](#4-demo-dashboard-quản-trị)
5. [Hệ Thống Thông Báo (Notifications)](#5-hệ-thống-thông-báo-notifications)
6. [Tài Liệu API Cho Ứng Dụng Khách (Customer)](#6-tài-liệu-api-cho-ứng-dụng-khách-customer)
7. [Di Chuyển Dữ Liệu Từ Ứng Dụng Cũ](#7-di-chuyển-dữ-liệu-từ-ứng-dụng-cũ)
8. [Kiến Trúc Kỹ Thuật](#8-kiến-trúc-kỹ-thuật)
9. [Xử Lý Sự Cố](#9-xử-lý-sự-cố)
10. [Demo Công Khai Qua Ngrok (Tuỳ Chọn)](#10-demo-công-khai-qua-ngrok-tuỳ-chọn)

---

## 1. Tổng Quan Kiến Trúc

```
                    ┌─────────────────────────┐
   Manager (bạn) ──▶│  Dashboard quản trị      │  (session login riêng, không dùng chung
                    │  /, /login, /manager/*   │   tài khoản với bất kỳ app khách nào)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐        ┌──────────────────────────┐
   App khách A ────▶│  clone-voice-station     │──HTTP─▶│  Colab (RVC + F5-TTS)     │
   (vd rag-legal-   │  /api/*  (X-Api-Key)     │        │  colab/voice_server.ipynb │
   assistant)       │  voice_station.db        │◀──HTTP─│  (train/convert giọng)    │
                    └───────────┬─────────────┘        └──────────────────────────┘
   App khách B ────▶            │  (webhook, khi manager xoá/vô hiệu hoá giọng)
   (công ty khác)    ◀──────────┘
```

- **Một service, nhiều "client"**: mỗi ứng dụng tích hợp (`rag-legal-assistant`, hoặc một công ty khác sau này) là một **client** riêng — có tên + API key riêng, dữ liệu (giọng nói, mẫu ghi âm) tách biệt hoàn toàn theo `client_id`.
- **Không dùng chung tài khoản người dùng**: clone-voice-station không biết username/password thật của người dùng cuối. Mỗi client tự định danh người dùng của mình bằng một chuỗi bất kỳ gọi là `external_user_id` (VD: `str(user_id)` phía `rag-legal-assistant`).
- **Manager dashboard** (mục 4) là **hoàn toàn tách biệt** — tài khoản quản trị (`managers` table) riêng, không liên quan gì tới tài khoản của client hay người dùng cuối. Đây là nơi vận hành viên của clone-voice-station quản lý toàn bộ hệ thống: cấu hình RVC, tạo/xoá client, xem và xoá/vô hiệu hoá bất kỳ giọng nói nào.

---

## 2. Yêu Cầu & Cài Đặt

```bash
pip install -r requirements.txt
```

Không cần Groq key hay bất kỳ LLM nào — service này chỉ xử lý giọng nói. Cần **ffmpeg** trong PATH (dùng bởi edge-TTS).

Không cần cấu hình gì thêm để chạy lần đầu — `voice_station.db`, client mặc định và tài khoản manager mặc định được tự tạo khi khởi động (mục 3).

---

## 3. Khởi Động Dịch Vụ

```bash
python app.py
```

Chạy trên **http://127.0.0.1:8090**. Lần đầu khởi động, terminal in ra (chỉ một lần):

```
[clone-voice-station] Seeded client 'rag-legal-assistant' — API key: <API_KEY>
[clone-voice-station] Put this key in rag-legal-assistant/voice_station_key.txt
[clone-voice-station] Seeded manager account — username: manager  password: <PASSWORD>
[clone-voice-station] Log in at http://127.0.0.1:8090/login and change this password.
```

**Lưu lại cả hai** — không có màn hình "quên mật khẩu": nếu mất mật khẩu manager, phải sửa trực tiếp trong `voice_station.db` (bảng `managers`) hoặc xoá dòng đó để service tự seed lại tài khoản mới ở lần khởi động kế tiếp.

> **Tài khoản manager hiện tại của máy này** (mật khẩu gốc lúc seed không kịp lưu lại trước khi mất, đã đặt lại thủ công qua `database.change_manager_password()`):
> - Username: `manager`
> - Password: `TestPass123!`
>
> Đổi lại bằng nút "Đổi mật khẩu" trong dashboard sau khi đăng nhập nếu muốn dùng mật khẩu khác.

---

## 4. Demo Dashboard Quản Trị

### Bước 1: Đăng Nhập

Truy cập **http://127.0.0.1:8090/login**, nhập username/password đã in ra ở mục 3. Sau khi đăng nhập, có thể đổi mật khẩu bằng nút **"Đổi mật khẩu"** ở góc trên bên phải dashboard (`POST /manager/change_password`).

### Bước 2: Kết Nối RVC (Colab)

Chạy notebook `colab/voice_server.ipynb` trên Google Colab, lấy URL tunnel (VD ngrok/cloudflared), dán vào ô **"Endpoint URL"** trong dashboard, nhấn **"Lưu endpoint"**. Chấm tròn cạnh tiêu đề chuyển **Online** nếu Colab đang chạy và phản hồi `/health`.

Có thể chỉnh **Pitch**, **Index rate** ngay bên dưới — áp dụng ngay cho lần convert giọng tiếp theo, không cần khởi động lại service. Mở rộng **"⚙️ Tuỳ chỉnh timeout nâng cao"** để chỉnh thời gian chờ cho từng loại request (convert/health-check/tải model/F5-TTS) nếu Colab phản hồi chậm hơn bình thường.

### Bước 3: Quản Lý Client Apps

Mục **"🧩 Client Apps"** liệt kê mọi ứng dụng đã tích hợp. Nhấn **"+ Tạo client"**, nhập tên (VD `acme-corp`) để cấp một API key mới — gửi key này cho đội kỹ thuật của ứng dụng đó (xem mục 6 để họ tích hợp).

- **Webhook**: mỗi client có thể đăng ký một URL để nhận thông báo khi manager xoá/vô hiệu hoá giọng nói của người dùng họ (mục 5). Client tự đăng ký qua API (`POST /api/webhook`, khuyến khích — tự động mỗi lần app khởi động), hoặc manager có thể tự đặt/xoá thủ công bằng nút **"Đặt webhook"** ở mỗi dòng.
- **Xoá client**: chỉ xoá được nếu client đó **chưa có giọng nói nào** đăng ký — tránh mất dữ liệu người dùng của họ một cách âm thầm. Xoá hết giọng nói của client đó trước (mục kế) rồi mới xoá được client.

### Bước 4: Quản Lý Toàn Bộ Giọng Nói

Mục **"🗂️ Toàn bộ giọng nói"** hiển thị mọi giọng nhân bản đã tạo, **bất kể thuộc client nào** — cột **Client** cho biết ứng dụng nào sở hữu. Có 3 thao tác mỗi dòng:

| Thao tác | Điều kiện | Ghi chú |
|---|---|---|
| Huấn luyện lại | Đủ mẫu ghi âm tối thiểu, không đang huấn luyện | Gửi lại yêu cầu train tới Colab |
| Vô hiệu hoá | Bất kỳ lúc nào | Chuyển trạng thái `failed`, giọng không dùng để đọc to được nữa, **nhưng không xoá dữ liệu mẫu** |
| Xoá | Bất kỳ lúc nào | Xoá vĩnh viễn: model trên Colab, file mẫu ghi âm, bản ghi database |

Cả **Vô hiệu hoá** và **Xoá** đều hỏi một **lý do tuỳ chọn** trước khi xác nhận — lý do này (nếu điền) được gửi kèm trong thông báo tới người dùng cuối (mục 5). **Không thể hoàn tác** thao tác Xoá.

---

## 5. Hệ Thống Thông Báo (Notifications)

Khi **manager** (không phải chính người dùng, không phải admin của app khách) xoá hoặc vô hiệu hoá một giọng nói, người dùng sở hữu giọng đó cần được biết — nếu không, họ sẽ thấy giọng "biến mất" mà không hiểu vì sao.

### 5.1 Luồng Xử Lý

```
Manager nhấn Xoá/Vô hiệu hoá (kèm lý do tuỳ chọn)
        ↓
clone-voice-station ghi 1 dòng vào bảng `notifications`
        ↓
Có webhook_url cho client này?
   ├─ Có  → POST tới webhook_url (kèm X-Api-Key của chính client đó)
   │         Thành công (HTTP 200) → đánh dấu delivered_at, xong
   │         Thất bại/timeout       → để nguyên chưa delivered (rơi xuống dưới)
   └─ Không hoặc thất bại → vẫn nằm trong GET /api/notifications
                            chờ client tự poll (fallback)
```

- **Webhook (đẩy tức thời)**: nếu client đã đăng ký `webhook_url`, thông báo được đẩy ngay lập tức khi manager thao tác — đây là cách `rag-legal-assistant` dùng (tự đăng ký `http://127.0.0.1:8000/voice/webhook` mỗi lần khởi động qua `voice/station_client.py`'s `register_webhook()`).
- **Polling (dự phòng)**: nếu chưa đăng ký webhook, hoặc app khách đang tắt lúc webhook được gửi, thông báo vẫn được lưu lại và trả về qua `GET /api/notifications?external_user_id=...` cho tới khi client gọi `POST /api/notifications/{id}/ack` xác nhận đã nhận — tránh mất thông báo, và tránh gửi lặp lại vô hạn.
- Xem mục 6.6 để biết cách một client tích hợp cả hai đường trên.

### 5.2 Trải Nghiệm Phía `rag-legal-assistant` (ví dụ tham khảo)

- Chuông 🔔 ở góc phải header trang chat, có số đếm thông báo chưa đọc.
- `GET /voice/notifications` vừa trả danh sách đã lưu cục bộ, vừa tự poll dự phòng từ clone-voice-station rồi nhập vào — nên dù webhook có lỡ thất bại, mở lại app vẫn thấy thông báo trong vòng tối đa 30 giây (chu kỳ poll của chuông).
- `POST /voice/webhook` (nhận đẩy từ clone-voice-station) xác thực bằng header `X-Api-Key` khớp với `voice_station_key.txt` của chính app đó — chỉ clone-voice-station (người giữ cùng key) mới gọi được.

---

## 6. Tài Liệu API Cho Ứng Dụng Khách (Customer)

Phần này dành cho đội kỹ thuật của một ứng dụng muốn **tích hợp tính năng giọng nói** thông qua clone-voice-station, thay vì tự xây dựng TTS/RVC riêng.

### 6.1 Xin Cấp API Key

Liên hệ manager của clone-voice-station để được tạo một **client** mới (mục 4, Bước 3) — nhận về một `api_key` duy nhất cho ứng dụng của bạn. Mọi request tới `/api/*` (trừ `/api/health`) đều cần header:

```
X-Api-Key: <api_key của bạn>
```

### 6.2 Khái Niệm `external_user_id`

clone-voice-station không có khái niệm tài khoản người dùng thật — bạn tự chọn một chuỗi định danh ổn định cho mỗi người dùng của mình (VD `str(user_id)` trong database của bạn, hoặc email) và luôn dùng đúng chuỗi đó cho mọi request liên quan tới người dùng ấy. Mỗi (`api_key` của bạn, `external_user_id`) là một namespace dữ liệu riêng — người dùng của client khác không thể thấy hay đụng tới dữ liệu của bạn.

### 6.3 Giọng Dựng Sẵn (Builtin)

Mọi client đều thấy chung 4 giọng dựng sẵn (không cần tạo, dùng ngay):

| Tên | `base_tts_voice` |
|---|---|
| HoaiMy (Nữ) | `vi-VN-HoaiMyNeural` |
| NamMinh (Nam) | `vi-VN-NamMinhNeural` |
| Jenny (EN) | `en-US-JennyNeural` |
| F5-TTS demo (VN) | `f5tts:default` |

### 6.4 Luồng Tích Hợp Cơ Bản (đọc to văn bản)

```bash
# 1. Lấy danh sách giọng khả dụng cho một người dùng (gồm builtin + giọng riêng của họ nếu có)
curl -H "X-Api-Key: $API_KEY" \
  "http://127.0.0.1:8090/api/profiles?external_user_id=user-42"

# 2. Đọc to một đoạn văn bản (bỏ trống profile_id để dùng giọng builtin mặc định)
curl -X POST -H "X-Api-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"text":"Xin chào, đây là bản demo giọng nói.","external_user_id":"user-42","profile_id":null}' \
  "http://127.0.0.1:8090/api/speak" \
  --output speech.mp3
```

### 6.5 Luồng Tích Hợp Nâng Cao (nhân bản giọng nói cá nhân)

```
1. POST /api/consent               — người dùng đồng ý điều khoản trước khi tạo giọng riêng
2. GET  /api/scripts                — lấy kịch bản đọc mẫu (8 đoạn văn, chủ đề đời thường)
3. POST /api/profiles                — tạo hồ sơ giọng mới (name), tối đa 2 giọng/người dùng
4. POST /api/profiles/{id}/samples   — upload từng đoạn ghi âm (multipart: script_id + audio)
   Lặp lại tối thiểu 5 lần (5 kịch bản khác nhau)
5. POST /api/profiles/{id}/train     — bắt đầu huấn luyện (chạy nền, không block)
6. GET  /api/profiles/{id}/status    — poll mỗi vài giây tới khi status="ready" (hoặc "failed")
7. POST /api/speak {..., "profile_id": <id>}  — dùng giọng vừa huấn luyện để đọc to
```

### 6.6 Đăng Ký Nhận Thông Báo (khuyến khích)

Để biết khi nào một giọng nói của người dùng bạn bị manager xoá/vô hiệu hoá (mục 5):

```bash
# Gọi lại mỗi lần app của bạn khởi động — chỉ là upsert, gọi lại nhiều lần vô hại
curl -X POST -H "X-Api-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"webhook_url":"https://your-app.example.com/voice/webhook"}' \
  "http://127.0.0.1:8090/api/webhook"
```

Endpoint của bạn (`/voice/webhook` ở ví dụ trên) cần:
- Chấp nhận `POST` với header `X-Api-Key: <chính api_key của bạn>` — dùng để xác thực request này thực sự đến từ clone-voice-station (chỉ hai bên biết key này).
- Đọc JSON body:
  ```json
  {
    "notification_id": 42,
    "event": "voice_profile_deleted",
    "external_user_id": "user-42",
    "profile_id": 7,
    "profile_name": "Giọng của tôi",
    "message": "Giọng nói \"Giọng của tôi\" của bạn đã bị quản trị viên xoá. Lý do: ..."
  }
  ```
  (`event` là `"voice_profile_deleted"` hoặc `"voice_profile_disabled"`.)
- Trả về HTTP 200 sau khi xử lý xong (VD lưu vào DB của bạn để hiển thị cho người dùng) — nếu không, clone-voice-station coi như gửi thất bại và giữ lại để bạn poll dự phòng.

**Dự phòng nếu chưa đăng ký webhook (hoặc webhook có lúc bị lỗi):**

```bash
# Gọi định kỳ (VD mỗi khi người dùng mở trang liên quan tới giọng nói)
curl -H "X-Api-Key: $API_KEY" \
  "http://127.0.0.1:8090/api/notifications?external_user_id=user-42"

# Sau khi đã xử lý xong mỗi thông báo (VD notification_id=42), xác nhận để không nhận lại:
curl -X POST -H "X-Api-Key: $API_KEY" \
  "http://127.0.0.1:8090/api/notifications/42/ack"
```

### 6.7 Bảng Tra Cứu API Đầy Đủ

Tất cả các endpoint dưới đây yêu cầu header `X-Api-Key`, trừ khi ghi chú khác.

| Endpoint | Method | Mô tả |
|---|---|---|
| `/api/health` | GET | Kiểm tra dịch vụ còn sống (không cần API key) |
| `/api/scripts` | GET | Kịch bản đọc mẫu để thu âm giọng riêng |
| `/api/consent` | GET | `?external_user_id=` — người dùng đã đồng ý điều khoản chưa |
| `/api/consent` | POST | `{external_user_id}` — ghi nhận đồng ý |
| `/api/profiles` | GET | `?external_user_id=` — danh sách giọng (builtin + của người dùng này) |
| `/api/profiles` | POST | `{external_user_id, name}` — tạo giọng riêng mới (cần đã đồng ý điều khoản) |
| `/api/profiles/{id}` | PUT | `{external_user_id, name?, is_default?}` — đổi tên / đặt mặc định |
| `/api/profiles/{id}` | DELETE | `?external_user_id=` — người dùng tự xoá giọng của mình |
| `/api/profiles/{id}/samples` | POST | multipart: `external_user_id, script_id, audio` — upload 1 mẫu ghi âm |
| `/api/profiles/{id}/samples` | GET | `?external_user_id=` — danh sách mẫu đã ghi |
| `/api/profiles/{id}/samples/{sample_id}` | DELETE | `?external_user_id=` — xoá 1 mẫu ghi âm |
| `/api/profiles/{id}/train` | POST | `{external_user_id}` — bắt đầu huấn luyện (chạy nền) |
| `/api/profiles/{id}/status` | GET | `?external_user_id=` — trạng thái huấn luyện |
| `/api/speak` | POST | `{external_user_id, text, profile_id}` — đọc to, trả về audio bytes |
| `/api/admin/voice_models` | GET | Mọi giọng riêng **của client bạn** (không thấy client khác) |
| `/api/admin/voice_models/{id}/retrain` | POST | Huấn luyện lại một giọng thuộc client bạn |
| `/api/admin/voice_models/{id}/disable` | POST | Vô hiệu hoá một giọng thuộc client bạn |
| `/api/admin/voice_models/{id}` | DELETE | Xoá một giọng thuộc client bạn |
| `/api/rvc_endpoint` | GET | Trạng thái/URL Colab hiện tại (dùng chung mọi client) |
| `/api/rvc_endpoint` | POST | `{endpoint}` — đổi URL Colab (ảnh hưởng **mọi** client — cân nhắc trước khi gọi) |
| `/api/webhook` | POST | `{webhook_url}` — đăng ký/cập nhật URL nhận thông báo (mục 6.6) |
| `/api/notifications` | GET | `?external_user_id=` — thông báo chưa xác nhận, dự phòng cho webhook |
| `/api/notifications/{id}/ack` | POST | Xác nhận đã nhận, không gửi lại nữa |

> **Lưu ý quan trọng:** `/api/admin/voice_models*` là API "admin của client bạn" (VD `rag-legal-assistant`'s trang `/admin/voice_models` gọi các route này) — khác với trang **dashboard quản trị của clone-voice-station** (mục 4), vốn dùng tài khoản manager riêng và có thể thấy/thao tác **mọi** client.

---

## 7. Di Chuyển Dữ Liệu Từ Ứng Dụng Cũ

Nếu một ứng dụng đã có sẵn dữ liệu giọng nói (VD `rag-legal-assistant` trước khi tách service này ra), dùng:

```bash
python migrate_from_rag_legal_assistant.py [đường-dẫn-tới-rag-legal-assistant]
```

Script này:
- Tạo (hoặc tái sử dụng) client `rag-legal-assistant` cùng API key.
- Copy mọi giọng **đã nhân bản** (`kind='cloned'`, không copy giọng builtin — service này tự seed builtin riêng) kèm mẫu ghi âm và model đã huấn luyện (nếu có bản backup cục bộ).
- Copy trạng thái đồng ý điều khoản (`voice_consent_at`) của từng người dùng.
- An toàn chạy lại nhiều lần — hồ sơ đã copy được đánh dấu để không copy trùng.

In ra API key cần dán vào `voice_station_key.txt` của ứng dụng nguồn sau khi chạy xong.

---

## 8. Kiến Trúc Kỹ Thuật

### 8.1 Cấu Trúc Thư Mục

```
clone-voice-station/
├── app.py                            # FastAPI app — dashboard (session) + /api/* (X-Api-Key)
├── database/
│   └── database.py                    # SQLite (voice_station.db): clients, voice_profiles,
│                                       #   voice_samples, voice_consent, settings, managers,
│                                       #   notifications
├── engine/
│   └── voice_engine.py                # speak_text() (TTS+RVC) / run_training() (RVC train)
├── voice/
│   ├── tts.py                         # edge-TTS + F5-TTS-Vietnamese-ViVoice (qua Colab)
│   ├── rvc_client.py                  # HTTP client gọi Colab (convert/train/health)
│   └── scripts.py                     # Kịch bản đọc mẫu để thu âm
├── colab/
│   └── voice_server.ipynb             # Notebook duy nhất: RVC + F5-TTS + STT Lab Tier 2 (train + serve)
├── templates/
│   ├── login.html                     # Đăng nhập manager
│   └── dashboard.html                 # Dashboard quản trị (mục 4)
├── voice_samples/                     # Mẫu ghi âm gốc (theo profile_id)
├── voice_storage/                     # Model RVC đã huấn luyện (backup cục bộ, theo speaker_id)
├── migrate_from_rag_legal_assistant.py
├── voice_station.db                   # SQLite — không commit lên git
├── session_secret.txt                 # Secret cho session cookie manager — không commit
└── requirements.txt
```

### 8.2 Schema Database

```
clients        (id, name UNIQUE, api_key UNIQUE, webhook_url, created_at)
voice_profiles (id, client_id, external_user_id, name, kind[builtin|cloned], base_tts_voice,
                speaker_id, status[new|collecting|training|ready|failed], is_default,
                error_message, model_local_path, created_at)
voice_samples  (id, profile_id, script_id, file_path, created_at)
voice_consent  (client_id, external_user_id, consented_at, PRIMARY KEY(client_id, external_user_id))
settings       (key, value)                     # rvc_endpoint, rvc_pitch, rvc_index_rate, rvc_timeout_*
managers       (id, username UNIQUE, password_hash, created_at)   # dashboard, PBKDF2-hashed
notifications  (id, client_id, external_user_id, profile_id, profile_name, event, message,
                created_at, delivered_at)        # mục 5
```

`voice_profiles.client_id`/`external_user_id` là `NULL` cho giọng builtin (dùng chung mọi client). Không có bảng nào lưu username/password thật của người dùng cuối — đó là trách nhiệm của app khách.

### 8.3 Vòng Đời Trạng Thái Giọng Nói (`voice_profiles.status`)

```
new ──(upload mẫu đầu tiên)──▶ collecting ──(POST .../train)──▶ training ──┬──▶ ready
                                                                            └──▶ failed
```

Manager có thể chuyển bất kỳ giọng nào sang `failed` bất cứ lúc nào (Vô hiệu hoá, mục 4 Bước 4) — giọng vẫn còn dữ liệu mẫu, có thể **Huấn luyện lại** để quay lại `ready`.

---

## 9. Xử Lý Sự Cố

### Manager quên mật khẩu

**Khắc phục:** Không có luồng quên mật khẩu tự động. Mở `voice_station.db`, xoá dòng tương ứng trong bảng `managers`, khởi động lại `python app.py` — service sẽ tự seed lại một tài khoản `manager` mới với mật khẩu ngẫu nhiên (in ra terminal, xem mục 3).

---

### RVC hiển thị "Offline" dù Colab đang chạy

**Kiểm tra:**
- URL endpoint trong dashboard đã đúng và không có dấu `/` thừa ở cuối
- Tunnel (ngrok/cloudflared) trên Colab còn sống — tunnel miễn phí thường hết hạn sau vài giờ, cần khởi động lại notebook và cập nhật URL mới
- Timeout "health/status" (mục 4, Tuỳ chỉnh nâng cao) đủ lớn nếu mạng chậm

---

### Client tích hợp không nhận được thông báo xoá/vô hiệu hoá

**Nguyên nhân có thể:**
- Chưa gọi `POST /api/webhook` để đăng ký URL (mục 6.6) — kiểm tra cột **Webhook** trong dashboard, mục Client Apps
- Endpoint webhook phía client trả về khác 200, hoặc timeout quá 8 giây — clone-voice-station sẽ không gửi lại tự động, thông báo nằm chờ ở trạng thái chưa nhận
- **Khắc phục không cần sửa gì thêm:** client cứ định kỳ gọi `GET /api/notifications?external_user_id=...` — mọi thông báo chưa `ack` đều được trả về, không mất dữ liệu, chỉ chậm hơn webhook

---

### Xoá client bị từ chối

**Thông báo:** `Client này còn N giọng nói — xoá hết trước khi xoá client.`

**Khắc phục:** Vào mục **"🗂️ Toàn bộ giọng nói"**, lọc theo cột Client, xoá từng giọng thuộc client đó trước, rồi quay lại xoá client.

---

### Huấn luyện giọng nói mãi không xong (`training` không chuyển `ready`)

**Kiểm tra:**
- Endpoint RVC còn online (mục RVC Connection trong dashboard)
- Đã ghi đủ tối thiểu 5 mẫu trước khi bấm huấn luyện
- Job tự dừng sau tối đa 2 giờ nếu Colab không phản hồi — cần khởi động lại Colab notebook rồi **Huấn luyện lại** (dashboard hoặc `POST /api/profiles/{id}/train`), không cần thu âm lại từ đầu

---

## 10. Demo Công Khai Qua Ngrok (Tuỳ Chọn)

> Tạm thời — dùng để demo/đánh giá thesis, chưa phải deploy thật (không có TLS/CDN/scaling, chỉ dựa vào auth sẵn có của từng app). Hướng future work là thay bằng deploy thật.

Có 2 chặng tunnel **độc lập nhau**, dùng cho 2 mục đích khác nhau:

1. **Colab (RVC) → clone-voice-station** — đã có UI sẵn trong dashboard (mục 2 ở trên: dán URL Colab vào ô "Endpoint URL"). **Không bắt buộc** phải làm bước 2/3 dưới đây mới demo được RVC — chặng này hoạt động độc lập.
2. **clone-voice-station → rag-legal-assistant / voice-lab-example** — chặng mới, dùng khi muốn cho người ngoài (không cùng máy) truy cập được 2 app khách này.

### Bước 1: Lộ clone-voice-station ra ngoài

```bash
set NGROK_AUTHTOKEN=your_token              # CMD -- lấy token tại https://dashboard.ngrok.com/tunnels/authtokens
$env:NGROK_AUTHTOKEN = "your_token"         # PowerShell
python start_ngrok.py
```

In ra một URL công khai (VD `https://xxxx.ngrok-free.app`) trỏ vào port 8090. Giữ cửa sổ này chạy — tunnel tắt khi đóng.

### Bước 2: Trỏ app khách vào URL đó

`rag-legal-assistant` và `voice-lab-example` đều đọc địa chỉ clone-voice-station qua biến môi trường `VOICE_STATION_URL` (`clone_voice_client`, xem `voice/station_client.py` phía rag-legal-assistant) — **chỉ set lúc khởi động, không có form nhập URL lúc đang chạy**, nên phải set biến rồi khởi động lại app đó:

```bash
set VOICE_STATION_URL=https://xxxx.ngrok-free.app          # CMD
$env:VOICE_STATION_URL = "https://xxxx.ngrok-free.app"     # PowerShell
python app.py
```

### Bước 3 (tuỳ chọn): Lộ luôn app khách đó ra ngoài

Nếu muốn người ngoài truy cập thẳng giao diện chat/demo của `rag-legal-assistant` hoặc `voice-lab-example` (không chỉ gọi API), chạy `start_ngrok.py` tương ứng trong thư mục app đó (port 8000 cho rag-legal-assistant, port 8091 cho voice-lab-example) — độc lập với tunnel ở Bước 1.

### Lưu ý

- Mỗi `start_ngrok.py` cần `NGROK_AUTHTOKEN` riêng (hoặc dùng chung 1 token cho nhiều tunnel nếu gói ngrok cho phép).
- Tunnel ngrok miễn phí đổi URL mỗi lần khởi động lại — cần lặp lại Bước 2 (và restart app khách) mỗi lần Bước 1 chạy lại.
- Không hardcode authtoken vào file — dùng biến môi trường (xem comment đầu mỗi `start_ngrok.py`).

---

## Ghi Chú Thêm

- Dữ liệu của mỗi client **tách biệt hoàn toàn** — client A không bao giờ thấy tên/giọng nói/thông báo của client B, kể cả qua API lẫn dashboard's cột Client (chỉ hiển thị cho manager, không lộ ra API của client khác).
- Giới hạn 2 giọng nhân bản/người dùng và tối thiểu 5 mẫu ghi âm là hằng số dùng chung cho mọi client (`MIN_TRAIN_SAMPLES`, `MAX_CLONED_VOICES_PER_USER` trong `database/database.py`) — hiện chưa cấu hình riêng theo từng client.
- `/api/rvc_endpoint` (đổi endpoint Colab) ảnh hưởng **toàn bộ** client cùng lúc vì chỉ có một kết nối Colab dùng chung — cân nhắc trước khi một client tự đổi endpoint qua API thay vì để manager làm qua dashboard.
