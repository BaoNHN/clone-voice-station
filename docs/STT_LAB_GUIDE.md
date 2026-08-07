# STT Lab & Voice Integration — Hướng dẫn kiến trúc & các luồng sử dụng

Tài liệu này mô tả toàn bộ hệ thống giọng nói được xây dựng trên 4 repo:

| Repo | Vai trò |
|---|---|
| [`clone-voice-station`](https://github.com/BaoNHN/clone-voice-station) | Service trung tâm — sở hữu training (RVC + STT), storage, và trang tự phục vụ STT Lab |
| [`clone-voice-client`](https://github.com/BaoNHN/clone-voice-client) | SDK Python mỏng — HTTP client + chế độ chạy Whisper cục bộ trong app |
| [`voice-lab-example`](https://github.com/BaoNHN/voice-lab-example) | App ví dụ độc lập — chứng minh chế độ "chạy trong thư viện app" hoạt động thật |
| [`rag-legal-assistant`](https://github.com/BaoNHN/rag-legal-assistant-voice) | Host app thật — trợ lý pháp lý RAG có tích hợp voice cloning (RVC) + mic live transcription |

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

## 9. Luồng RVC voice cloning (`rag-legal-assistant`)

Đây là tính năng "nhân bản giọng nói" chính của thesis 24MSE23204 — khác hoàn toàn với STT Lab (STT Lab là nhận dạng giọng nói *đầu vào*; RVC là tạo giọng nói *đầu ra* mô phỏng giọng người dùng). Vào `/voice` (đã đăng nhập):

1. **Đồng ý xử lý dữ liệu sinh trắc học** — tích checkbox trên `disclaimerCard` → `POST /voice/consent`. Nội dung gated (`gatedContent`) chỉ mở khi `session_info` trả về `voice_consent: true`.
2. **Tạo hồ sơ giọng nói** — đặt tên → `POST /voice/profiles`.
3. **Ghi mẫu** — mở màn ghi âm cho hồ sơ đó (`openRecording`), đọc lần lượt các đoạn kịch bản có sẵn (`GET /voice/scripts`, xem `voice/scripts.py` phía `clone-voice-station`), bấm ghi/dừng cho từng đoạn (`toggleRecord`) → mỗi đoạn tải lên `POST /voice/profiles/{id}/samples`.
4. **Huấn luyện** — nút "🚀 Bắt đầu huấn luyện" chỉ bật khi đã ghi đủ tối thiểu `MIN_SAMPLES` (mặc định 5) đoạn khác nhau → `POST /voice/profiles/{id}/train`. Trang tự poll `GET /voice/profiles/{id}/status` định kỳ, hiện `progress_message` trực tiếp (vd "Đang trích xuất đặc trưng HuBERT trên cpu…") thay vì chỉ một pill trạng thái tĩnh.
5. **Dùng giọng đã train** — khi `status = ready`, có thể đặt làm mặc định (`PUT /voice/profiles/{id}` với `is_default`) rồi chọn trong dropdown giọng nói ở trang chat (`templates/index.html`) để nghe câu trả lời bằng giọng đã nhân bản (`POST /voice/speak`).
6. **Phân biệt "chưa có giọng" với "dịch vụ đang down"** — cả `index.html` lẫn `voice_profile.html` gọi `GET /voice/status` (health-check `clone-voice-station`) khi danh sách hồ sơ rỗng, thay vì mặc định coi rỗng = "chưa có giọng nào" (trước đây 2 trường hợp không phân biệt được).

Việc train thật sự (RVC v2: preprocess → F0 RMVPE → HuBERT → train → FAISS index) chạy trên Colab qua `colab/voice_server.ipynb`, xem mục 11 bên dưới.

## 10. Luồng mic live transcription (hỏi đáp bằng giọng nói)

Trên trang chat (`rag-legal-assistant`), nút mic bên cạnh ô nhập câu hỏi:

1. Bấm mic → trình duyệt xin quyền micro → `MediaRecorder` bắt đầu ghi theo chunk 500ms (`_mediaRecorder.start(500)`).
2. **Trong lúc vẫn đang ghi**, mỗi 2.5 giây (`LIVE_TRANSCRIBE_INTERVAL_MS`) đoạn ghi âm tích luỹ được gửi lại `POST /voice/transcribe` → text hiển thị dần trong ô nhập câu hỏi, cập nhật theo thời gian thực trong lúc người dùng vẫn đang nói. Nếu người dùng tự gõ gì đó vào ô nhập giữa chừng, live-tick sẽ không ghi đè lên (so sánh `chatInput.value` với transcript live gần nhất).
3. Bấm mic lần nữa để dừng ghi → gửi lần transcribe **cuối cùng, có giá trị quyết định** (dùng toàn bộ audio đã ghi, không chỉ đoạn mới) → điền vào ô nhập.
4. Không tự động gửi câu hỏi — Whisper có thể nghe nhầm thuật ngữ pháp lý, người dùng cần cơ hội xem/sửa trước khi bấm Gửi.

Toàn bộ luồng này chỉ dùng **remote mode** (gọi `clone-voice-station` qua HTTP) — chưa chuyển sang local mode của `clone-voice-client` (điều đó chỉ có trong `voice-lab-example`, mục 6).

## 11. Chạy Colab notebook

`clone-voice-station/colab/` có 3 notebook. Cả 3 đều đã được sửa để không còn phụ thuộc `rvc-python`/fairseq (thư viện này không build được trên Python 3.11+, phiên bản runtime Colab hiện tại) — xem mục "Giới hạn/lịch sử" cuối file.

### `voice_server.ipynb` — bản đang dùng thật (production)

Notebook duy nhất phục vụ cả RVC **và** STT Lab Tier 2, chạy tuần tự từ trên xuống:

1. Mount Google Drive → clone RVC-Project repo + cài dependency (đã lọc bớt, không cần `rvc-python`) → kiểm tra GPU → tải model pretrained V2 (Generator/Discriminator/HuBERT/RMVPE từ HuggingFace).
2. Cell định nghĩa `train_speaker()` (pipeline RVC đầy đủ) + hàng đợi training RVC riêng (`_train_queue`, 1 job/lần).
3. Cell `!pip install -q peft` + `train_stt_adapter()` (LoRA fine-tune Whisper, Tier 2) + hàng đợi STT riêng (`_stt_train_queue`) — tách biệt khỏi hàng đợi RVC dù dùng chung 1 GPU.
4. (Tuỳ chọn) F5-TTS-Vietnamese-ViVoice baseline, PhoWhisper cho `/transcribe`.
5. Flask server khởi động — expose `/health /models /train /train_status/<id> /convert` (RVC) và `/stt_train /stt_train_status/<id> /stt_train/<id>/download` (STT Tier 2).
6. Cell cloudflared tunnel — in ra URL dạng `https://xxxx.trycloudflare.com`.
7. **Dán URL đó vào manager dashboard của `clone-voice-station`** (`/manager` → RVC endpoint) — cùng một endpoint được dùng cho cả RVC lẫn STT Lab Tier 2 khi guest chọn backend "Colab"/"Auto".

Notebook phải giữ chạy liên tục — dừng/khởi động lại runtime sẽ đổi URL tunnel, phải dán lại.

### `train_rvc.ipynb` / `serve_rvc.ipynb` — bản cũ, tách rời huấn luyện/phục vụ

Hai notebook này có trước khi mọi thứ được gộp vào `voice_server.ipynb`; vẫn hữu ích khi muốn train một giọng cụ thể theo kịch bản thủ công (đúng tinh thần thesis Section 4.1 Stage 2), tách biệt khỏi service đang chạy:

**`train_rvc.ipynb`**: sửa `SPEAKER_NAME` trong cell CONFIGURATION → mount Drive → tải audio thô lên `DATASET_RAW` trên Drive → chạy tuần tự các cell còn lại (slice/normalize → preprocess → F0 → HuBERT → filelist → train ~200 epoch, 30–60 phút trên T4 → build FAISS index → export `.pth`+`.index` vào `MODELS_DIR` trên Drive).

**`serve_rvc.ipynb`**: chạy sau khi đã có `.pth`+`.index` từ `train_rvc.ipynb` (cùng `SPEAKER_NAME`) → mount Drive + clone RVC repo + cài dependency → kiểm tra GPU → khởi động Flask server (`/health`, `/convert`, shell ra `infer/cli.py` của repo thay vì gọi thẳng `rvc-python`) → cloudflared tunnel → in `RVC_ENDPOINT` để dán vào `.env`/biến môi trường của app.

## 12. Giới hạn đã biết / ngoài phạm vi

- Chưa có preview WER trước/sau khi train.
- Upload mẫu chỉ qua file, chưa có ghi âm trực tiếp trong `/stt-lab` (chỉ `voice-lab-example` có mic UI).
- `voice-lab-example` chưa demo Tier 2 LoRA (chỉ có sẵn hạ tầng, chưa nối UI/luồng thử).
- Mic live transcription trong `rag-legal-assistant` chỉ dùng remote mode — chưa chuyển sang local mode của `clone-voice-client`.
- `train_rvc.ipynb`/`serve_rvc.ipynb` là bản cũ (tách rời), không có route STT Lab Tier 2 — chỉ `voice_server.ipynb` có.
