

## Overfitting

**Định nghĩa**: Overfitting (quá khớp) là hiện tượng model học quá chi tiết và ghi nhớ từng mẫu cụ thể trong training data, thay vì học các pattern tổng quát. Điều này khiến model hoạt động tốt trên dữ liệu training nhưng kém trên dữ liệu mới (validation/test set).

**Nguyên nhân chính**: Model có quá nhiều tham số so với số lượng dữ liệu training, cho phép nó "ghi nhớ" thay vì "học hỏi".

Với 100 samples, model có thể học thuộc lòng (memorize) thay vì học pattern. Điều này xảy ra khi:

- **Dataset quá nhỏ**: Với chỉ 100 mẫu, model có đủ capacity để ghi nhớ từng mẫu cụ thể thay vì học các pattern tổng quát
- **Không có khả năng tổng quát hóa**: Model có thể đạt accuracy cao trên training set nhưng performance kém trên validation/test set
- **Dấu hiệu nhận biết**: 
  - Training loss giảm nhanh và về gần 0
  - Validation loss không giảm hoặc tăng dần
  - Model không thể xử lý các input tương tự nhưng khác một chút so với training data

**Giải pháp**:
- Tăng số lượng samples trong dataset
- Sử dụng data augmentation
- Áp dụng regularization techniques (dropout, weight decay)
- Early stopping dựa trên validation loss
- Sử dụng cross-validation để đánh giá tốt hơn

## Downstream

**Định nghĩa**: Downstream tasks (tác vụ downstream) là các tác vụ cụ thể mà model được fine-tune để thực hiện sau khi đã được pre-train trên dữ liệu tổng quát.

**Giải thích chi tiết**:
- **Pre-training**: Model được train trên dữ liệu lớn, tổng quát (như Wikipedia, Common Crawl) để học ngôn ngữ nói chung
- **Downstream tasks**: Các tác vụ cụ thể như:
  - **Text classification**: Phân loại văn bản (sentiment analysis, spam detection)
  - **Question Answering**: Trả lời câu hỏi
  - **Named Entity Recognition (NER)**: Nhận diện thực thể có tên
  - **Machine Translation**: Dịch máy
  - **Text Summarization**: Tóm tắt văn bản
  - **Chatbot/Dialogue**: Hệ thống hội thoại

**Tại sao gọi là "downstream"?**
- Thuật ngữ này xuất phát từ khái niệm "upstream" (pre-training) và "downstream" (fine-tuning)
- Pre-training là bước đầu tiên (upstream), tạo ra model có kiến thức tổng quát
- Fine-tuning là bước tiếp theo (downstream), điều chỉnh model cho tác vụ cụ thể

**Ví dụ**: 
- Model GPT được pre-train trên internet text (upstream)
- Sau đó fine-tune cho tác vụ phân loại cảm xúc (downstream task)
- Hoặc fine-tune cho chatbot hỗ trợ khách hàng (downstream task)

**Lợi ích của PEFT với downstream tasks**:
- Fine-tune nhanh hơn: Chỉ cần train một phần nhỏ tham số thay vì toàn bộ model
- Tiết kiệm tài nguyên: Không cần GPU mạnh như full fine-tuning
- Dễ dàng thử nghiệm: Có thể fine-tune cho nhiều downstream tasks khác nhau mà không cần lưu toàn bộ model cho mỗi task

## Lớp Attention (Attention Layer)

**Định nghĩa**: Lớp attention là một thành phần quan trọng trong kiến trúc Transformer, cho phép model tập trung vào các phần quan trọng của input khi xử lý từng token.

**Cách hoạt động**:
- **Self-Attention**: Mỗi token trong sequence có thể "chú ý" đến tất cả các token khác (bao gồm chính nó) để hiểu mối quan hệ giữa chúng
- **Query, Key, Value (QKV)**: 
  - **Query (Q)**: Đại diện cho token hiện tại đang được xử lý
  - **Key (K)**: Đại diện cho các token khác trong sequence
  - **Value (V)**: Chứa thông tin thực tế của các token
- **Attention Score**: Tính toán mức độ liên quan giữa Query và Key để quyết định trọng số cho mỗi Value

**Ví dụ**: 
- Trong câu "The cat sat on the mat", khi xử lý từ "sat", attention layer sẽ tập trung vào "cat" (chủ ngữ) và "mat" (vị trí) để hiểu ngữ cảnh đầy đủ.

**Vai trò trong Transformer**:
- Cho phép model hiểu mối quan hệ dài hạn giữa các từ trong câu
- Giúp model xử lý được context phức tạp và phụ thuộc xa
- Là thành phần chính tạo nên sức mạnh của các LLM hiện đại

## Lớp Fully-Connected (Fully-Connected Layer / Feed-Forward Network)

**Định nghĩa**: Lớp fully-connected (hay còn gọi là Feed-Forward Network - FFN) là một mạng neural đơn giản áp dụng cho mỗi token một cách độc lập, thường nằm sau lớp attention trong Transformer.

**Cấu trúc**:
- **Input**: Nhận đầu vào từ lớp attention
- **Hidden layers**: Thường có 2 lớp linear với activation function (như ReLU, GELU) ở giữa
- **Output**: Trả về representation mới của token sau khi được xử lý

**Công thức cơ bản**:
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
hoặc với GELU:
```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**Vai trò**:
- **Xử lý thông tin cục bộ**: Áp dụng phép biến đổi phi tuyến cho mỗi token
- **Mở rộng không gian biểu diễn**: Tăng khả năng biểu diễn của model
- **Kết hợp với Attention**: 
  - Attention layer: Xử lý mối quan hệ giữa các token (tương tác ngang)
  - FFN layer: Xử lý thông tin của từng token (xử lý dọc)

**Trong kiến trúc Transformer**:
- Mỗi Transformer block thường có:
  1. **Multi-Head Self-Attention** (lớp attention)
  2. **Feed-Forward Network** (lớp fully-connected)
  3. **Layer Normalization** và **Residual Connection** xung quanh mỗi thành phần

**Ví dụ trong adapter-based methods**:
- Adapter được chèn sau lớp attention và sau lớp fully-connected
- Cho phép model học các điều chỉnh cụ thể cho task mà không thay đổi các tham số gốc của pretrained model

## CAUSAL_LM (Causal Language Model)

**Định nghĩa**: CAUSAL_LM (Causal Language Model) là một loại language model dự đoán token tiếp theo dựa trên các token trước đó, chỉ sử dụng thông tin từ quá khứ (left-to-right).

**Đặc điểm chính**:
- **Unidirectional (một chiều)**: Model chỉ nhìn về phía trước (từ trái sang phải)
- **Autoregressive (tự hồi quy)**: Mỗi token được dự đoán dựa trên tất cả các token trước nó
- **Masked attention**: Sử dụng causal mask để đảm bảo token tại vị trí i chỉ có thể "nhìn thấy" các token từ vị trí 0 đến i-1

**Cách hoạt động**:
- Khi xử lý sequence `[x₁, x₂, x₃, ..., xₙ]`:
  - Token x₁: Không có context (hoặc chỉ có special token như `<BOS>`)
  - Token x₂: Chỉ thấy x₁
  - Token x₃: Thấy x₁ và x₂
  - Token xₙ: Thấy tất cả các token từ x₁ đến xₙ₋₁

**Causal Mask**:
```
[1, 0, 0, 0]  ← Token 1 chỉ thấy chính nó
[1, 1, 0, 0]  ← Token 2 thấy token 1 và 2
[1, 1, 1, 0]  ← Token 3 thấy token 1, 2, và 3
[1, 1, 1, 1]  ← Token 4 thấy tất cả
```

**Ví dụ**:
- Input: "The cat sat on"
- Model dự đoán: "the" (token tiếp theo)
- Khi generate: "The cat sat on the mat"
- Model tiếp tục dự đoán token tiếp theo dựa trên toàn bộ context trước đó

**Ứng dụng**:
- **Text Generation**: Tạo văn bản tự động (GPT, GPT-2, GPT-3, GPT-4)
- **Chatbot**: Hội thoại tự nhiên
- **Code Completion**: Hoàn thiện code
- **Story Writing**: Viết truyện, bài viết

**Trong Hugging Face Transformers**:
- `model_type = "causal_lm"` hoặc `AutoModelForCausalLM`
- Các model phổ biến: GPT-2, GPT-Neo, GPT-J, LLaMA, Mistral, Phi
- Task type: `TaskType.CAUSAL_LM`

**So sánh với các loại Language Model khác**:
- **CAUSAL_LM** (GPT-style): Unidirectional, chỉ nhìn về quá khứ
- **MASKED_LM** (BERT-style): Bidirectional, nhìn cả quá khứ và tương lai (với mask)
- **SEQ_2_SEQ_LM** (T5-style): Encoder-Decoder, có thể nhìn cả hai phía

**Ưu điểm**:
- Phù hợp cho text generation
- Tự nhiên trong việc tạo văn bản tuần tự
- Có thể generate text dài một cách hiệu quả

**Nhược điểm**:
- Không thể sử dụng thông tin từ tương lai (như BERT)
- Có thể bỏ lỡ context quan trọng ở cuối câu khi xử lý token đầu tiên

**Trong PEFT**:
- CAUSAL_LM thường được sử dụng với các phương pháp như LoRA, Adapter, Prefix Tuning
- Cho phép fine-tune các model như GPT, LLaMA cho các downstream tasks cụ thể