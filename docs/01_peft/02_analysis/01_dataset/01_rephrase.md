# Phân Tích Dataset Rephrase

## Tổng Quan

Dataset này được sử dụng để fine-tune model nhằm tạo ra cấu trúc JSON output từ input query. Model sẽ học cách:
- Tối ưu hóa query thành keyword phù hợp cho search engine
- Xác định xem query có trong phạm vi trả lời hay không (`is_in_scope`)
- Đưa ra lý do cho quyết định (`reasoning`)
- Tạo message banner thông báo cho khách hàng trong thời gian chờ search engine
- Tạo message dự phòng khi không có kết quả (`message_no_result`)

## Cấu Trúc Dữ Liệu

### Input
- **query**: Câu truy vấn từ người dùng (string)

### Output (JSON Structure)
```json
{
  "keyword": "string",           // Query tối ưu cho search engine
  "is_in_scope": boolean,        // Trong phạm vi trả lời hay không
  "reasoning": "string",         // Lý do cho is_in_scope
  "message_banner": "string",    // Message thông báo trong thời gian chờ
  "message_no_result": "string"   // Message dự phòng khi không có kết quả
}
```

### Metadata
- **status**: Trạng thái (0 = thành công)
- **count**: Số lần xuất hiện
- **prompt_version**: Phiên bản prompt được sử dụng
- **key**: Hash key của query
- **vector_distance**: Khoảng cách vector (độ tương đồng)

## Thống Kê Tổng Quan

### Số Lượng Samples
- **Tổng số samples**: 1,000
- **Samples `is_in_scope=True`**: 961 (96.10%)
- **Samples `is_in_scope=False`**: 39 (3.90%)
- **Samples có `reasoning`**: 40 (4.00%)

### Phân Phối Độ Dài

| Trường | Trung Bình | Min | Max | Median |
|--------|-----------|-----|-----|--------|
| **query** | 15.72 | 1 | 74 | 14.00 |
| **keyword** (non-empty) | 25.40 | 6 | 59 | 25.00 |
| **message_banner** (non-empty) | 103.94 | 66 | 161 | - |
| **message_no_result** (non-empty) | 122.85 | 77 | 178 | - |

## Phân Tích Chi Tiết

### 1. Phân Tích `is_in_scope`

#### In-Scope Samples (961 samples - 96.10%)
- ✅ **Tất cả đều có `keyword`**: 961/961 (100%)
- ✅ **Tất cả đều có `message_banner`**: 961/961 (100%)
- ⚠️ **Có `reasoning`**: 1/961 (0.10%) - Rất ít

**Đặc điểm**:
- Tất cả samples in-scope đều có đầy đủ keyword và message_banner
- Hầu hết không có reasoning (chỉ 1 sample có)
- Keyword được tối ưu hóa từ query gốc
- Message banner thân thiện, có emoji, nhấn mạnh từ khóa bằng `<b>`

#### Out-of-Scope Samples (39 samples - 3.90%)
- ❌ **Không có `keyword`**: 0/39 (0%)
- ❌ **Không có `message_banner`**: 0/39 (0%)
- ✅ **Có `reasoning`**: 39/39 (100%)

**Đặc điểm**:
- Tất cả đều có reasoning giải thích tại sao không trong phạm vi
- Không có keyword và message_banner
- Có message_no_result để thông báo cho người dùng

### 2. Phân Tích `reasoning`

**Tổng số samples có reasoning**: 40
- **Số loại reasoning khác nhau**: 22
- **Out-of-scope samples**: 39/40 (97.5%)
- **In-scope samples**: 1/40 (2.5%)

**Top 5 lý do phổ biến nhất**:
1. "Từ khóa không liên quan đến sản phẩm mẹ và bé tại Con Cưng." (6 lần)
2. "Mã sản phẩm không rõ, không liên quan đến hệ sinh thái Con Cưng." (4 lần)
3. "Truy vấn không rõ ràng, không liên quan đến sản phẩm mẹ và bé." (3 lần)
4. "Sản phẩm không thuộc hệ sinh thái mẹ và bé của Con Cưng." (3 lần)
5. "Sản phẩm không liên quan đến mẹ và bé trong hệ sinh thái Con Cưng." (3 lần)

**Các loại lý do chính**:
- Không liên quan đến sản phẩm mẹ và bé
- Mã sản phẩm không rõ
- Truy vấn không rõ ràng
- Không thuộc hệ sinh thái Con Cưng
- Sản phẩm không phù hợp (đồ gia dụng, snack, v.v.)

### 3. Phân Tích Query

**Thống kê từ khóa**:
- **Tổng số từ**: 3,577
- **Số từ unique**: 1,331
- **Tỷ lệ từ unique**: 37.2%

**Top 20 từ phổ biến nhất**:
1. "sữa" (205 lần) - 5.7%
2. "bé" (85 lần) - 2.4%
3. "cho" (84 lần) - 2.3%
4. "grow" (37 lần) - 1.0%
5. "bình" (33 lần) - 0.9%
6. "sinh" (33 lần) - 0.9%
7. "tã" (32 lần) - 0.9%
8. "tả" (32 lần) - 0.9%
9. "1" (31 lần) - 0.9%
10. "quần" (29 lần) - 0.8%
11. "trai" (29 lần) - 0.8%
12. "bỉm" (26 lần) - 0.7%
13. "đồ" (25 lần) - 0.7%
14. "bột" (24 lần) - 0.7%
15. "tháng" (24 lần) - 0.7%
16. "size" (23 lần) - 0.6%
17. "sơ" (23 lần) - 0.6%
18. "nước" (23 lần) - 0.6%
19. "kem" (22 lần) - 0.6%
20. "dán" (22 lần) - 0.6%

**Nhận xét**:
- Dataset tập trung vào sản phẩm mẹ và bé (sữa, tã, bỉm, đồ cho bé)
- Có nhiều từ khóa liên quan đến độ tuổi (tháng, size)
- Có cả từ tiếng Việt và tiếng Anh (grow, size)
- Có các biến thể chính tả (tã/tả)

### 4. Phân Tích Metadata

**Status**:
- Tất cả samples đều có `status = 0` (thành công)

**Count** (số lần xuất hiện):
- **Min**: 1
- **Max**: 55
- **Trung bình**: 6.43
- **Median**: 5

**Vector Distance** (độ tương đồng):
- **Min**: 0.0000
- **Max**: 0.6099
- **Trung bình**: 0.4042
- **Median**: 0.4210

**Nhận xét**:
- Vector distance phân phối đều, không có outliers rõ ràng
- Count cho thấy một số query xuất hiện nhiều lần (có thể là query phổ biến)

## Phân Tích Chất Lượng Dữ Liệu

### Điểm Mạnh

1. **Tỷ lệ cân bằng hợp lý**:
   - 96.1% in-scope vs 3.9% out-of-scope
   - Đủ dữ liệu để model học cả hai trường hợp

2. **Dữ liệu đầy đủ**:
   - Tất cả in-scope samples đều có keyword và message_banner
   - Tất cả out-of-scope samples đều có reasoning

3. **Độ dài hợp lý**:
   - Query ngắn gọn (trung bình 15.72 ký tự)
   - Keyword được tối ưu (trung bình 25.40 ký tự)
   - Message có độ dài phù hợp cho UX

4. **Đa dạng từ khóa**:
   - 1,331 từ unique trong 3,577 từ tổng cộng
   - Bao phủ nhiều chủ đề sản phẩm mẹ và bé

### Điểm Yếu và Cần Cải Thiện

1. **Thiếu reasoning cho in-scope**:
   - Chỉ 1/961 in-scope samples có reasoning
   - Model sẽ khó học cách tạo reasoning cho in-scope cases

2. **Số lượng out-of-scope ít**:
   - Chỉ 39 samples (3.9%)
   - Có thể không đủ để model học tốt việc phân loại out-of-scope

3. **Reasoning không đa dạng**:
   - Chỉ 22 loại reasoning khác nhau
   - Nhiều reasoning trùng lặp

4. **Thiếu edge cases**:
   - Cần thêm các trường hợp biên (query rất ngắn, rất dài, có ký tự đặc biệt)

## Đặc Điểm Dataset Cần Lưu Ý

### 1. Cấu Trúc JSON Output
- Tất cả output đều là JSON structure với 5 trường: `keyword`, `is_in_scope`, `reasoning`, `message_banner`, `message_no_result`
- Output phải là valid JSON với escape characters đúng

### 2. Pattern Empty Fields
- **In-scope samples**: `reasoning` thường rỗng (chỉ 1/961 có reasoning)
- **Out-of-scope samples**: `keyword` và `message_banner` luôn rỗng (0/39 có)

### 3. HTML Tags và Emoji
- `message_banner` và `message_no_result` chứa HTML tags (`<b>`, `</b>`) để nhấn mạnh từ khóa
- Các message thường có emoji để tạo cảm giác thân thiện

### 4. Tiếng Việt
- Toàn bộ dataset là tiếng Việt
- Có một số từ tiếng Anh (brand names, technical terms)
- Có biến thể chính tả (tã/tả)

### 5. Tính Nhất Quán
- Keyword thường liên quan đến query gốc nhưng được tối ưu hóa
- Message banner và message_no_result có tone và style nhất quán (thân thiện, hỗ trợ)
- Tất cả message đều bắt đầu với "Con Cưng" hoặc "Ba mẹ"

## Tài Liệu Tham Khảo

- Dataset file: `src/lora/dataset/01_simple/01_dataset_rephrase.json`
- PEFT Methods: `docs/01_peft/01_research/`
- Training Guide: `docs/01_peft/02_analysis/02_peft/`

