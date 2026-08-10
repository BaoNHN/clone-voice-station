"""
voice/scripts.py
Fixed set of Vietnamese reading paragraphs used to collect voice samples for
training a personal cloned voice (see templates/voice_profile.html).

Paragraphs are general-topic (not legal text) and phonetically varied —
covering different vowels, tones and consonant clusters — so a handful of
short recordings still give the RVC training pipeline reasonably diverse
audio. Each paragraph was trimmed to roughly half its original length
(~50-60 words, ~15-20s spoken) so a single recording is quicker to get
through without dropping below RVC's practical per-clip minimum. Reading
all 8 now gives ~5-6 minutes total, short of the "~10-15 min of clean
speech" guidance in colab/voice_server.ipynb — if trained-voice quality
suffers, prefer asking users for extra takes (re-reading the same
paragraph makes a new sample) or adding more paragraphs over lengthening
these back out.
"""

READING_SCRIPTS = [
    {
        "id": "p1",
        "title": "Thời tiết",
        "text": (
            "Hôm nay trời có nắng nhẹ vào buổi sáng, nhưng đến chiều mây đen kéo đến "
            "và có thể sẽ có mưa rào. Nhiệt độ dao động trong khoảng hai mươi lăm đến "
            "ba mươi hai độ C. Người dân được khuyến cáo mang theo áo mưa khi ra đường, "
            "đặc biệt là ở khu vực miền Trung."
        ),
    },
    {
        "id": "p2",
        "title": "Ẩm thực",
        "text": (
            "Phở là một trong những món ăn nổi tiếng nhất của Việt Nam, được nấu từ "
            "nước dùng xương hầm trong nhiều giờ cùng với các loại gia vị như quế, hồi, "
            "gừng nướng và hành tây. Bánh phở được trụng qua nước sôi rồi cho thịt bò "
            "hoặc thịt gà lên trên, thêm hành lá, rau thơm và một chút chanh, ớt tùy "
            "khẩu vị từng người."
        ),
    },
    {
        "id": "p3",
        "title": "Du lịch",
        "text": (
            "Vịnh Hạ Long là một trong những kỳ quan thiên nhiên nổi tiếng thế giới, "
            "với hàng nghìn hòn đảo đá vôi nhô lên giữa làn nước xanh biếc. Du khách "
            "thường chọn đi thuyền để khám phá các hang động kỳ vĩ, chèo thuyền kayak "
            "quanh những vách đá dựng đứng, hoặc đơn giản là ngắm hoàng hôn buông xuống "
            "trên mặt biển tĩnh lặng."
        ),
    },
    {
        "id": "p4",
        "title": "Gia đình",
        "text": (
            "Vào mỗi dịp cuối tuần, cả nhà tôi thường quây quần bên nhau để cùng nấu "
            "một bữa cơm thật ấm cúng. Ông bà kể chuyện ngày xưa, bố mẹ chia sẻ công "
            "việc trong tuần, còn lũ trẻ thì háo hức khoe những điểm số tốt ở trường. "
            "Sau bữa ăn, mọi người cùng nhau xem một bộ phim hoặc chơi vài ván cờ."
        ),
    },
    {
        "id": "p5",
        "title": "Công nghệ",
        "text": (
            "Trí tuệ nhân tạo đang dần thay đổi cách con người làm việc và học tập "
            "trong nhiều lĩnh vực khác nhau, từ y tế, giáo dục cho đến sản xuất công "
            "nghiệp. Các trợ lý ảo có thể trả lời câu hỏi, dịch ngôn ngữ, hay thậm chí "
            "sáng tác nhạc chỉ trong vài giây."
        ),
    },
    {
        "id": "p6",
        "title": "Thành phố",
        "text": (
            "Buổi sáng ở thành phố luôn nhộn nhịp với dòng người hối hả đi làm, tiếng "
            "còi xe vang lên khắp các con phố. Những gánh hàng rong bán xôi, bánh mì "
            "hay cà phê sữa đá đã trở thành một phần không thể thiếu trong nhịp sống "
            "đô thị."
        ),
    },
    {
        "id": "p7",
        "title": "Giáo dục",
        "text": (
            "Học tập suốt đời là một triết lý ngày càng được nhiều người áp dụng trong "
            "thời đại thay đổi nhanh chóng như hiện nay. Không chỉ dừng lại ở trường "
            "lớp, kiến thức có thể đến từ sách vở, các khóa học trực tuyến, hay đơn "
            "giản là từ những cuộc trò chuyện với người có kinh nghiệm."
        ),
    },
    {
        "id": "p8",
        "title": "Sở thích",
        "text": (
            "Vào những ngày rảnh rỗi, tôi thường thích đọc sách bên cửa sổ, nghe một "
            "bản nhạc nhẹ nhàng, hoặc chăm sóc mấy chậu cây nhỏ trên ban công. Thỉnh "
            "thoảng, tôi cùng bạn bè đi leo núi vào cuối tuần để hít thở không khí "
            "trong lành và ngắm nhìn khung cảnh thiên nhiên hùng vĩ từ trên cao."
        ),
    },
]


def get_scripts():
    return READING_SCRIPTS


def get_script(script_id: str):
    return next((s for s in READING_SCRIPTS if s["id"] == script_id), None)
