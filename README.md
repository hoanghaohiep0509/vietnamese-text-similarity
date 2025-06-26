# Khảo sát Độ Tương Đồng Văn Bản Tiếng Việt

Dự án nghiên cứu và đánh giá các phương pháp đo độ tương đồng văn bản tiếng Việt, sử dụng nhiều phương pháp tokenize và độ đo khác nhau.

## Tính năng chính

- So sánh độ tương đồng giữa hai văn bản tiếng Việt
- Hỗ trợ nhiều phương pháp tách từ:
  - Underthesea
  - RDRSegmenter
  - Pyvi
  - VnCoreNLP

- Đa dạng độ đo tương đồng:
  - Độ đo dựa trên chuỗi (String-based):
    - Cosine Similarity
    - Jaccard Similarity
    - Levenshtein Distance
    - Longest Common Subsequence
  - Độ đo dựa trên tập hợp (Set-based):
    - Overlap Coefficient
    - Dice Coefficient
  - Độ đo dựa trên vector (Vector-based):
    - TF-IDF Cosine Similarity
    - Word2Vec Similarity
  - Độ đo ngữ nghĩa (Semantic-based):
    - PhoBERT Embedding Similarity


- Giao diện web thân thiện
- Hỗ trợ dataset mẫu với nhiều cặp văn bản đã gán nhãn
- Biểu đồ so sánh kết quả các phương pháp
- API RESTful cho tích hợp hệ thống

## Yêu cầu hệ thống

- Python 3.8+
- Java Runtime Environment (JRE) cho VnCoreNLP
- Pip package manager
- Node.js và npm (cho frontend)

## Cài đặt

1. Clone repository:

```bash

git clone https://github.com/hoanghaohiep0509/vietnamese-text-similarity.git
cd vietnamese-text-similarity
```

2. Cài đặt dependencies:

```bash
# Cài đặt Python packages
pip install -r requirements.txt

# Cài đặt VnCoreNLP
bash setup.sh
```

3. Khởi động ứng dụng:

```bash
# Chạy backend
python backend/main.py

# Mở frontend
# Có thể sử dụng Live Server hoặc HTTP server đơn giản
python -m http.server 8080
```

## Cấu trúc thư mục

# vietnamese-text-similarity
vietnamese-text-similarity/
├── backend/
│ ├── main.py # Entry point của backend
│ ├── preprocessing/ # Xử lý tiền xử lý văn bản
│ ├── similarity_measures/ # Các độ đo tương đồng
│ └── utils/ # Tiện ích
├── frontend/
│ ├── index.html # Giao diện người dùng
│ ├── styles.css # CSS styles
│ └── scripts.js # JavaScript
├── data/
│ └── test_dataset/ # Dataset mẫu
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
├── setup.sh # Script cài đặt
└── README.md


## Sử dụng

1. Truy cập giao diện web:
   - Backend API: `http://localhost:5001`
   - Frontend: `http://localhost:8080`

2. Chọn phương thức nhập văn bản:
   - Nhập trực tiếp hai văn bản
   - Chọn từ dataset có sẵn

3. Cấu hình tham số:
   - Chọn phương pháp tách từ
   - Chọn độ đo tương đồng
   - Tùy chỉnh các tham số khác (nếu có)

4. Xem kết quả:
   - Điểm số tương đồng
   - Thời gian xử lý
   - Biểu đồ so sánh (nếu chọn nhiều phương pháp)

## API Documentation

### Endpoints

1. `GET /api/health`
   - Kiểm tra trạng thái hoạt động của API

2. `POST /api/similarity`
   - Tính độ tương đồng giữa hai văn bản
   ```json
   {
     "text1": "Văn bản thứ nhất",
     "text2": "Văn bản thứ hai",
     "similarity_method": "cosine_tfidf",
     "tokenize_method": "underthesea"
   }
   ```

3. `GET /api/methods`
   - Lấy danh sách các phương pháp có sẵn

4. `GET /api/dataset`
   - Lấy dataset mẫu

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Liên hệ

Hoang Hao Hiep - 519H0160

Project Link: [https://github.com/hoanghaohiep0509/vietnamese-text-similarity.git](https://github.com/hoanghaohiep0509/vietnamese-text-similarity.git)

## Acknowledgments

* [Underthesea](https://github.com/undertheseanlp/underthesea)
* [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP)
* [PhoBERT](https://github.com/VinAIResearch/PhoBERT)