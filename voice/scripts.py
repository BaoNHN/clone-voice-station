"""
voice/scripts.py
Fixed set of Vietnamese reading paragraphs used to collect voice samples for
training a personal cloned voice (see templates/voice_profile.html).

Paragraphs are general-topic (not legal text) and phonetically varied —
covering different vowels, tones and consonant clusters — so a handful of
short recordings still give the RVC training pipeline reasonably diverse
audio. Reading all of them gives ~10-12 minutes of speech, matching the
"~10-15 min of clean speech" guidance in colab/voice_server.ipynb.
"""

READING_SCRIPTS = [
    {
        "id": "p1",
        "title": "Thời tiết",
        "text": (
            "Hôm nay trời có nắng nhẹ vào buổi sáng, nhưng đến chiều mây đen kéo đến "
            "và có thể sẽ có mưa rào. Nhiệt độ dao động trong khoảng hai mươi lăm đến "
            "ba mươi hai độ C. Người dân được khuyến cáo mang theo áo mưa khi ra đường, "
            "đặc biệt là ở khu vực miền Trung, nơi thường xuyên chịu ảnh hưởng của gió "
            "mùa vào thời điểm này trong năm. Vào buổi tối, nhiệt độ sẽ giảm xuống và "
            "trời trở nên mát mẻ hơn, thích hợp cho các hoạt động ngoài trời như đi dạo "
            "hoặc tập thể dục nhẹ nhàng cùng gia đình và bạn bè."
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
            "khẩu vị từng người. Mỗi vùng miền lại có một cách chế biến phở riêng, tạo "
            "nên sự phong phú và đa dạng cho nền ẩm thực nước nhà, khiến du khách quốc "
            "tế luôn muốn quay lại thưởng thức thêm lần nữa."
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
            "trên mặt biển tĩnh lặng. Nhiều gia đình lựa chọn nghỉ đêm trên du thuyền để "
            "trải nghiệm trọn vẹn vẻ đẹp huyền bí của vịnh vào lúc bình minh, khi sương "
            "mù còn giăng nhẹ trên những đỉnh núi đá."
        ),
    },
    {
        "id": "p4",
        "title": "Gia đình",
        "text": (
            "Vào mỗi dịp cuối tuần, cả nhà tôi thường quây quần bên nhau để cùng nấu "
            "một bữa cơm thật ấm cúng. Ông bà kể chuyện ngày xưa, bố mẹ chia sẻ công "
            "việc trong tuần, còn lũ trẻ thì háo hức khoe những điểm số tốt ở trường. "
            "Sau bữa ăn, mọi người cùng nhau xem một bộ phim hoặc chơi vài ván cờ, "
            "tiếng cười nói vang khắp căn nhà nhỏ. Những khoảnh khắc giản dị như vậy "
            "luôn là điều quý giá nhất, giúp gắn kết các thế hệ trong gia đình lại gần "
            "nhau hơn dù cuộc sống có bận rộn đến đâu."
        ),
    },
    {
        "id": "p5",
        "title": "Công nghệ",
        "text": (
            "Trí tuệ nhân tạo đang dần thay đổi cách con người làm việc và học tập "
            "trong nhiều lĩnh vực khác nhau, từ y tế, giáo dục cho đến sản xuất công "
            "nghiệp. Các trợ lý ảo có thể trả lời câu hỏi, dịch ngôn ngữ, hay thậm chí "
            "sáng tác nhạc chỉ trong vài giây. Tuy nhiên, việc ứng dụng công nghệ mới "
            "cũng đặt ra không ít thách thức về đạo đức, quyền riêng tư và an toàn dữ "
            "liệu cá nhân, đòi hỏi các nhà phát triển phải cân nhắc kỹ lưỡng trước khi "
            "đưa sản phẩm đến tay người dùng cuối cùng."
        ),
    },
    {
        "id": "p6",
        "title": "Thành phố",
        "text": (
            "Buổi sáng ở thành phố luôn nhộn nhịp với dòng người hối hả đi làm, tiếng "
            "còi xe vang lên khắp các con phố. Những gánh hàng rong bán xôi, bánh mì "
            "hay cà phê sữa đá đã trở thành một phần không thể thiếu trong nhịp sống "
            "đô thị. Khi đêm xuống, ánh đèn neon rực rỡ thắp sáng các khu phố mua sắm, "
            "thu hút đông đảo giới trẻ đến vui chơi, chụp ảnh và thưởng thức ẩm thực "
            "đường phố đa dạng, tạo nên bức tranh sống động của một đô thị không bao "
            "giờ ngủ."
        ),
    },
    {
        "id": "p7",
        "title": "Giáo dục",
        "text": (
            "Học tập suốt đời là một triết lý ngày càng được nhiều người áp dụng trong "
            "thời đại thay đổi nhanh chóng như hiện nay. Không chỉ dừng lại ở trường "
            "lớp, kiến thức có thể đến từ sách vở, các khóa học trực tuyến, hay đơn "
            "giản là từ những cuộc trò chuyện với người có kinh nghiệm. Việc rèn luyện "
            "tư duy phản biện và khả năng tự học sẽ giúp mỗi cá nhân thích nghi tốt hơn "
            "trước những biến động của thị trường lao động, đồng thời mở ra nhiều cơ "
            "hội phát triển sự nghiệp trong tương lai."
        ),
    },
    {
        "id": "p8",
        "title": "Sở thích",
        "text": (
            "Vào những ngày rảnh rỗi, tôi thường thích đọc sách bên cửa sổ, nghe một "
            "bản nhạc nhẹ nhàng, hoặc chăm sóc mấy chậu cây nhỏ trên ban công. Thỉnh "
            "thoảng, tôi cùng bạn bè đi leo núi vào cuối tuần để hít thở không khí "
            "trong lành và ngắm nhìn khung cảnh thiên nhiên hùng vĩ từ trên cao. Những "
            "sở thích tưởng chừng đơn giản ấy lại giúp tôi cân bằng cuộc sống, giảm bớt "
            "căng thẳng sau những giờ làm việc mệt mỏi, và tìm lại niềm vui trong những "
            "điều nhỏ bé thường ngày."
        ),
    },
]


def get_scripts():
    return READING_SCRIPTS


def get_script(script_id: str):
    return next((s for s in READING_SCRIPTS if s["id"] == script_id), None)
