# Đánh Giá Dataset Rephrase cho Retrain Llama 8B

## 1. Tổng Quan Dataset

### 1.1. Thống Kê Cơ Bản
- **Tổng số mẫu**: 100 samples
- **Samples có keyword**: 97 samples (97%)
- **Samples in-scope**: 97 samples (97%)
- **Samples out-of-scope**: 3 samples (3%)
- **Độ dài query trung bình**: 17.3 ký tự
- **Độ dài keyword trung bình**: 25.0 ký tự

### 1.2. Cấu Trúc Dữ Liệu
Mỗi sample chứa:
- `query`: Câu truy vấn gốc từ người dùng (input)
- `output.keyword`: Từ khóa đã được tối ưu hóa cho search (output)
- `output.is_in_scope`: Boolean xác định query có liên quan đến domain
- `output.reasoning`: Lý do xử lý (nếu có)
- Metadata khác: `status`, `count`, `vector_distance`, `prompt_version`

## 2. Phân Tích Chất Lượng Dataset

### 2.1. Điểm Mạnh

#### ✅ Đa Dạng Về Loại Query
Dataset bao phủ nhiều loại query khác nhau:
- **Query có lỗi chính tả**: "dau goi be" → "dầu gội cho bé", "co rửa" → "cọ rửa"
- **Query thiếu thông tin**: "nhổ" → "dụng cụ nhổ răng cho bé"
- **Query không chuẩn**: "biỉm" → "bỉm cho bé", "biu tính tiền" → "máy tính tiền đồ chơi cho bé"
- **Query cần chuẩn hóa**: "ngủ cốc" → "ngũ cốc", "tã đan" → "tã dán"
- **Query cần mở rộng**: "tia lưỡi" → "rơ lưỡi cho bé"
- **Query cần tinh chỉnh**: "combo dầu tràm hoàng cung" → "combo dầu tràm Cung Đình"

#### ✅ Dữ Liệu Thực Tế
- Query đến từ người dùng thực tế với các lỗi phổ biến
- Phản ánh cách người dùng tìm kiếm trong thực tế
- Bao gồm cả edge cases (query không hợp lệ, noise)

#### ✅ Mapping Rõ Ràng
- Input-output mapping rõ ràng và nhất quán
- Keyword được tối ưu hóa cho mục đích search
- Có metadata hỗ trợ (is_in_scope, reasoning)

### 2.2. Điểm Yếu

#### ⚠️ Kích Thước Dataset Nhỏ
- **100 samples** là khá nhỏ cho fine-tuning Llama 8B
- Thông thường cần ít nhất 500-1000+ samples cho task cụ thể
- Có thể dẫn đến overfitting hoặc model không generalize tốt

#### ⚠️ Mất Cân Bằng Về Độ Dài
- Query ngắn (trung bình 17.3 ký tự)
- Keyword dài hơn (trung bình 25.0 ký tự)
- Có thể cần thêm samples với query dài hơn để model học tốt hơn

#### ⚠️ Thiếu Đa Dạng Về Domain
- Tập trung chủ yếu vào domain "mẹ và bé"
- Có thể cần mở rộng sang các domain khác nếu muốn model general hơn

#### ⚠️ Edge Cases Chưa Đủ
- Chỉ có 3 samples out-of-scope
- Cần thêm nhiều edge cases để model học cách xử lý query không hợp lệ

## 3. Đánh Giá Khả Thi Cho Retrain Llama 8B

### 3.1. Khả Thi ✅

#### Lý Do Khả Thi:
1. **Task Rõ Ràng**: Rephrase query → optimized keyword là task cụ thể, phù hợp với fine-tuning
2. **Format Phù Hợp**: Input-output mapping rõ ràng, dễ format cho training
3. **Domain-Specific**: Dataset tập trung vào domain cụ thể, phù hợp cho fine-tuning
4. **Chất Lượng Tốt**: Mapping chính xác, có logic rõ ràng

### 3.2. Rủi Ro Và Hạn Chế ⚠️

#### Rủi Ro:
1. **Overfitting**: Với 100 samples, model có thể học thuộc lòng thay vì học pattern
2. **Generalization Kém**: Model có thể không xử lý tốt các query ngoài training set
3. **Bias Domain**: Model có thể quá tập trung vào domain "mẹ và bé"

#### Hạn Chế:
1. **Cần Thêm Dữ Liệu**: Nên mở rộng dataset lên ít nhất 500-1000 samples
2. **Cần Validation Set**: Cần tách riêng validation set để đánh giá
3. **Cần Test Set**: Cần test set với các query chưa từng thấy

## 4. Khuyến Nghị

### 4.1. Trước Khi Retrain

#### ✅ Nên Làm:
1. **Mở Rộng Dataset**:
   - Thu thập thêm ít nhất 400-900 samples nữa
   - Đảm bảo đa dạng về loại query (typo, thiếu thông tin, không chuẩn, etc.)
   - Thêm edge cases (out-of-scope queries)

2. **Chuẩn Bị Dữ Liệu**:
   - Tách train/validation/test (80/10/10 hoặc 70/15/15)
   - Format dữ liệu theo chuẩn instruction tuning
   - Thêm prompt template phù hợp

3. **Đánh Giá Baseline**:
   - Test với base model Llama 8B trước khi fine-tune
   - So sánh với các phương pháp khác (rule-based, embedding search, etc.)

### 4.2. Trong Quá Trình Retrain

#### ✅ Nên Làm:
1. **Sử Dụng LoRA/QLoRA**:
   - Fine-tune với LoRA để giảm chi phí và tránh overfitting
   - QLoRA nếu thiếu GPU memory

2. **Hyperparameter Tuning**:
   - Learning rate: 1e-4 đến 5e-4
   - Batch size: 4-8
   - Epochs: 3-5 (monitor để tránh overfitting)
   - LoRA rank: 8-16

3. **Monitoring**:
   - Track loss trên train và validation set
   - Early stopping nếu validation loss tăng
   - Đánh giá định kỳ trên test set

### 4.3. Sau Khi Retrain

#### ✅ Nên Làm:
1. **Đánh Giá Model**:
   - Test trên test set chưa từng thấy
   - Đánh giá trên các loại query khác nhau
   - So sánh với baseline và các phương pháp khác

2. **A/B Testing**:
   - Deploy và so sánh với hệ thống hiện tại
   - Thu thập feedback từ người dùng
   - Iterate và cải thiện

## 5. Format Dữ Liệu Đề Xuất

### 5.1. Instruction Format
```
### Instruction:
Hãy chuyển đổi query người dùng thành từ khóa tối ưu cho tìm kiếm sản phẩm.

### Input:
{query}

### Output:
{keyword}
```

### 5.2. Ví Dụ
```
### Instruction:
Hãy chuyển đổi query người dùng thành từ khóa tối ưu cho tìm kiếm sản phẩm.

### Input:
dau goi be

### Output:
dầu gội cho bé
```

## 6. Kết Luận

### 6.1. Tổng Kết
- ✅ **Dataset có khả thi** cho retrain Llama 8B cho chức năng rephrase
- ⚠️ **Cần mở rộng dataset** từ 100 lên ít nhất 500-1000 samples
- ✅ **Chất lượng mapping tốt**, logic rõ ràng
- ⚠️ **Cần cẩn thận với overfitting** do dataset nhỏ

### 6.2. Đánh Giá Tổng Thể
**Điểm số: 7/10**

- **Chất lượng dữ liệu**: 8/10 (tốt nhưng cần thêm)
- **Kích thước dataset**: 4/10 (quá nhỏ)
- **Đa dạng**: 7/10 (tốt nhưng tập trung một domain)
- **Khả thi**: 8/10 (có thể làm được với điều kiện mở rộng dataset)

### 6.3. Quyết Định
**CÓ THỂ TIẾN HÀNH** với điều kiện:
1. Mở rộng dataset lên ít nhất 500 samples
2. Sử dụng LoRA/QLoRA để fine-tune
3. Có validation và test set riêng
4. Monitor cẩn thận để tránh overfitting

