# Test độ đo
from backend.similarity_measures.set_based import jaccard_similarity

# Lấy một cặp văn bản từ dataset.csv
text1 = "Việt Nam thắng trận bóng đá lần thứ 1 với tỷ số ngẫu nhiên."
text2 = "Đội tuyển Việt Nam giành chiến thắng trận bóng đá thứ 1."

similarity = jaccard_similarity(text1, text2)
print(f"Độ tương đồng Jaccard: {similarity:.3f}")