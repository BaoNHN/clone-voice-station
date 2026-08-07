# STT Lab — Hướng dẫn kiến trúc & sử dụng

Tài liệu này mô tả toàn bộ hệ thống giọng nói (STT) được xây dựng trên 3 repo:

| Repo | Vai trò |
|---|---|
| [`clone-voice-station`](https://github.com/BaoNHN/clone-voice-station) | Service trung tâm — sở hữu training, storage, và trang tự phục vụ STT Lab |
| [`clone-voice-client`](https://github.com/BaoNHN/clone-voice-clien) | SDK Python mỏng — HTTP client + chế độ chạy Whisper cục bộ trong app |
| [`voice-lab-example`](https://github.com/BaoNHN/voice-lab-example) | App ví dụ độc lập — chứng minh chế độ "chạy trong thư viện app" hoạt động thật |

## 1. Vì sao có 3 repo tách biệt

`clone-voice-client` không nằm trong `clone-voice-station` vì nó là SDK dùng chung cho *bất kỳ* host app nào (không chỉ voice-station), và version của nó không nên khoá theo nhịp release của server. `voice-lab-example` cũng không nằm trong `clone-voice-client` vì nó chỉ là một app tiêu thụ SDK đó, không phải một phần của bản thân SDK.

## 2. Hai chế độ chạy STT

- **Remote (lazy-load)** — hành vi gốc: host app gọi `POST /api/transcribe` trên `clone-voice-station`, Whisper chạy trên server, lazy-load lần gọi đầu tiên.
- **Local (nhúng thư viện)** — mới thêm: `clone-voice-client[local]` chạy Whisper (và LoRA adapter nếu có) ngay trong process của host app, không gọi qua mạng tới voice-station. Đây là điều `voice-lab-example` chứng minh.

Hai chế độ **cộng thêm**, không thay thế nhau — remote vẫn là mặc định không đổi.

## 3. STT Lab — trang tự phục vụ cho khách (`/stt-lab`)

Khác với luồng RVC (voice cloning) vốn cần API key từ host app, STT Lab cho khách **tự đăng ký tài khoản trực tiếp** (username/password, không API key) tại `clone-voice-station`'s `/stt-lab`. Vì trang công khai không có cổng chặn, mọi giới hạn tài nguyên (số adapter/khách, số mẫu/adapter, thời lượng audio, hàng đợi train cục bộ) đều được enforce chặt hơn so với luồng RVC.

### Tier 1 — Hotword / prompt-bias (không cần GPU)

Khách tạo adapter, nhập danh sách từ khoá/thuật ngữ riêng (vd thuật ngữ pháp lý). Không cần huấn luyện — dùng ngay để bias Whisper qua `initial_prompt`. Tải về `.stt-pack.zip` chứa `manifest.json` + `hotwords.json`.

### Tier 2 — LoRA fine-tune thật (cần GPU)

Khách upload cặp (audio, transcript đúng), chọn base model (`whisper-tiny`/`whisper-base` — giới hạn vì VRAM máy chủ chỉ ~4GB và trang không có cổng chặn), rồi huấn luyện qua 1 trong 3 lựa chọn backend:

- **Auto** — thử Colab trước, tự động chuyển sang cục bộ nếu Colab không khả dụng (giống RVC).
- **Colab** — bắt buộc Colab, KHÔNG âm thầm chuyển sang cục bộ nếu thất bại (để so sánh trung thực 2 phương pháp).
- **Local** — luôn chạy trên máy chủ.

Job cục bộ đi qua **hàng đợi 1-job-tại-1-thời-điểm** (`engine/stt_train_engine.py`) — khác với RVC hiện tại (không có hàng đợi cho local fallback, một lỗ hổng có thật được phát hiện khi khảo sát, không lặp lại ở đây vì STT Lab công khai hơn RVC).

Tải về `.stt-pack.zip` (khi đã train xong) chứa thêm `adapter_model.safetensors` + `adapter_config.json` — dùng được thật qua `clone_voice_client.local_stt`.

### Train tiếp từ pack cũ

Khách có thể upload lại `.stt-pack.zip` đã tải trước đó (`POST /api/stt/adapters/{id}/continue`) để train tiếp từ đúng weight cũ thay vì train lại từ đầu — vì data thô ban đầu đã bị xoá theo cam kết privacy, chỉ weight nhỏ được giữ lại.

### Xoá dữ liệu

Khách có thể xoá từng adapter hoặc toàn bộ tài khoản bất cứ lúc nào — xoá thật cả DB lẫn file audio/model trên đĩa, không chỉ vô hiệu hoá.

## 4. Colab vs Local — so sánh 2 kiến trúc training

| | Colab | Local |
|---|---|---|
| GPU | GPU miễn phí của Colab | GPU/CPU của máy chủ (~4GB VRAM) |
| Hàng đợi | `queue.Queue()` riêng trong notebook (`colab/voice_server.ipynb`) | `queue.Queue()` riêng trong `engine/stt_train_engine.py` |
| Chia sẻ tài nguyên | Cùng GPU với training RVC — 2 hàng đợi tách biệt, có thể tranh chấp nếu chạy cùng lúc | Độc lập với RVC |
| Khi ép buộc mà không khả dụng | Báo lỗi rõ ràng, không tự chuyển | (không áp dụng — local luôn khả dụng nếu máy chạy được) |

## 5. `clone-voice-client` — dùng trong host app

```python
from clone_voice_client import VoiceStationClient

client = VoiceStationClient(base_url="http://127.0.0.1:8090", api_key="...")

# Remote (mặc định, không đổi)
result = client.transcribe(filename, audio_bytes, mime="audio/webm", language="vi")

# Local — cần: pip install clone-voice-client[local]
from clone_voice_client import local_stt
pack = local_stt.load_pack("adapter.stt-pack.zip")   # tự nhận diện Tier 1 hay Tier 2
result = client.transcribe_local(filename, audio_bytes, mime="audio/webm",
                                  language="vi", pack=pack)
```

`load_pack()` trả về `{"tier", "base_model", "hotwords", "adapter_dir"}` — `adapter_dir` chỉ có giá trị khi pack là Tier 2 (chứa LoRA weight thật), lúc đó `transcribe_local()` tự động dùng `transformers` + `peft` thay vì `openai-whisper` (PEFT chỉ gắn được vào model class của HF, không gắn được vào `openai-whisper`).

## 6. `voice-lab-example` — chạy thử

```bash
cd voice-lab-example
pip install -r requirements.txt      # kéo theo clone-voice-client[local] (nặng: torch+transformers+peft)
python app.py                        # http://127.0.0.1:8091
```

Chat dùng retrieval TF-IDF thật trên bộ dữ liệu công khai `MrCookieDev/Vietnamese-Chatting-Dataset` (không phải câu trả lời cứng). Nút mic ghi âm → `POST /transcribe` → Whisper chạy ngay trong process của app này (không gọi `clone-voice-station` qua mạng).

Muốn thấy hotword/LoRA áp dụng thật: chạy `clone-voice-station`, vào `/stt-lab`, tạo adapter, tải `.stt-pack.zip`, thả vào `voice-lab-example/stt_pack/`, khởi động lại.

## 7. API tham khảo nhanh (`clone-voice-station`)

```
POST   /stt-lab/register              đăng ký khách (username/password)
POST   /stt-lab/login
POST   /stt-lab/logout
GET    /stt-lab                       trang dashboard khách

POST   /api/stt/adapters              tạo adapter (name, base_model)
GET    /api/stt/adapters              danh sách adapter của khách
GET    /api/stt/adapters/{id}         chi tiết 1 adapter (poll status)
PUT    /api/stt/adapters/{id}         đổi tên / cập nhật hotwords
DELETE /api/stt/adapters/{id}         xoá adapter (+ file trên đĩa)
GET    /api/stt/adapters/{id}/download  tải .stt-pack.zip

POST   /api/stt/adapters/{id}/samples          upload mẫu (audio, transcript)
GET    /api/stt/adapters/{id}/samples          danh sách mẫu
DELETE /api/stt/adapters/{id}/samples/{sid}    xoá 1 mẫu

POST   /api/stt/adapters/{id}/train      {"backend": "auto"|"colab"|"local"}
POST   /api/stt/adapters/{id}/continue   upload pack cũ để train tiếp

DELETE /api/stt/account               xoá toàn bộ tài khoản + dữ liệu
```

## 8. Giới hạn đã biết / ngoài phạm vi

- Chưa có preview WER trước/sau khi train.
- Upload mẫu chỉ qua file, chưa có ghi âm trực tiếp trong `/stt-lab` (chỉ `voice-lab-example` có mic UI).
- `voice-lab-example` chưa demo Tier 2 LoRA (chỉ có sẵn hạ tầng, chưa nối UI/luồng thử).
- Chưa tích hợp voice interaction vào ứng dụng chat chính (`rag-legal-assistant`) — đây vẫn là tính năng độc lập, minh hoạ qua `voice-lab-example`.
