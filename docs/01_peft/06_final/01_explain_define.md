

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


## Qwen 7B với 4-bit quantization

**Định nghĩa**: Qwen 7B là một Large Language Model (LLM) có 7 tỷ tham số được phát triển bởi Alibaba Cloud. 4-bit quantization là kỹ thuật giảm độ chính xác của các trọng số (weights) từ 16-bit (float16) hoặc 32-bit (float32) xuống 4-bit để giảm đáng kể bộ nhớ GPU cần thiết.

### Qwen 7B Model

**Thông tin cơ bản**:
- **Kiến trúc**: Transformer-based decoder-only (tương tự GPT)
- **Số tham số**: 7 tỷ (7B) parameters
- **Hidden size**: 4096
- **Number of layers**: 28
- **Number of attention heads**: 32
- **Context length**: 32,768 tokens
- **Precision gốc**: float16 hoặc bfloat16

**Kích thước model**:
- **Không quantization**: ~14 GB (với float16)
- **Với 4-bit quantization**: ~4-5 GB
- **Giảm**: ~65-70% bộ nhớ

### 4-bit Quantization

**Định nghĩa**: Quantization là quá trình chuyển đổi các giá trị số từ độ chính xác cao (high precision) sang độ chính xác thấp (low precision) để giảm bộ nhớ và tăng tốc độ inference.

**Cách hoạt động**:
1. **Chuyển đổi weights**: 
   - Từ float16 (16-bit, 65,536 giá trị có thể) → int4 (4-bit, 16 giá trị có thể)
   - Mỗi weight được ánh xạ vào một trong 16 giá trị có thể (quantization levels)

2. **Quantization Scheme**:
   - **NF4 (NormalFloat4)**: Sử dụng phân phối chuẩn hóa, tối ưu cho weights của neural networks
   - **FP4 (FloatPoint4)**: Sử dụng floating point representation
   - **BitsAndBytes**: Thư viện thực hiện quantization hiệu quả

3. **Dequantization**:
   - Khi inference, weights được dequantize về float16 để tính toán
   - Quá trình này diễn ra tự động trong quá trình forward pass

**Công thức cơ bản**:
```
Quantized Value = round((Original Value - Zero Point) / Scale)
Dequantized Value = Quantized Value × Scale + Zero Point
```

### Tại sao sử dụng 4-bit Quantization?

**Vấn đề với Qwen 7B không quantization**:
- Cần ~14 GB VRAM chỉ để load model
- Hầu hết consumer GPUs (RTX 3060, RTX 4060 Ti) chỉ có 12-16 GB VRAM
- Không đủ memory để chạy model + activation memory + generation

**Lợi ích của 4-bit quantization**:
1. **Giảm bộ nhớ đáng kể**:
   - Từ ~14 GB → ~4-5 GB (giảm ~65-70%)
   - Cho phép chạy trên GPU nhỏ hơn (12 GB VRAM)

2. **Tăng tốc độ inference**:
   - Ít data cần load từ memory
   - Có thể cache nhiều hơn trong GPU memory

3. **Tiết kiệm chi phí**:
   - Không cần GPU cao cấp (A100, H100)
   - Có thể chạy trên consumer GPUs

4. **Giữ được chất lượng**:
   - Với NF4 quantization, chất lượng giảm rất ít (< 5% trong hầu hết tasks)
   - Vẫn đủ tốt cho hầu hết các ứng dụng

### Cấu hình BitsAndBytesConfig

**Ví dụ cấu hình**:
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Sử dụng 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16, # Compute dtype (float16 cho tốc độ)
    bnb_4bit_use_double_quant=True,       # Double quantization (tiết kiệm thêm)
    bnb_4bit_quant_type="nf4"             # Loại quantization (NF4)
)
```

**Giải thích các tham số**:
- **`load_in_4bit=True`**: Load model với 4-bit quantization
- **`bnb_4bit_compute_dtype`**: Dtype để tính toán (float16 hoặc bfloat16)
- **`bnb_4bit_use_double_quant=True`**: Áp dụng quantization 2 lần cho quantization constants, tiết kiệm thêm ~0.4 GB
- **`bnb_4bit_quant_type="nf4"`**: Sử dụng NF4 quantization scheme (tối ưu nhất)

### So sánh Memory Usage

| Component | Không Quantization | 4-bit Quantization | Giảm |
|-----------|-------------------|-------------------|------|
| Model Weights | ~14 GB | ~4-5 GB | ~65-70% |
| Activation Memory | Phụ thuộc input length | Phụ thuộc input length | Không đổi |
| KV Cache | Phụ thuộc input length | Phụ thuộc input length | Không đổi |
| **Total (với input ngắn)** | ~14-15 GB | ~5-6 GB | ~60% |

**Lưu ý**: Quantization chỉ giảm memory cho model weights, không giảm activation memory hoặc KV cache.

### Nhược điểm

1. **Chất lượng giảm nhẹ**:
   - Có thể giảm 2-5% accuracy trong một số tasks
   - Với NF4, chất lượng vẫn rất tốt

2. **Tốc độ inference**:
   - Có thể chậm hơn một chút do cần dequantize
   - Nhưng thường nhanh hơn do ít memory bandwidth hơn

3. **Không thể fine-tune trực tiếp**:
   - Quantized weights không thể train trực tiếp
   - Cần sử dụng PEFT (LoRA, Adapter) để fine-tune

4. **Compatibility**:
   - Cần thư viện `bitsandbytes` được cài đặt đúng
   - Một số GPU cũ có thể không hỗ trợ

### Khi nào sử dụng 4-bit Quantization?

**Nên sử dụng khi**:
- GPU có VRAM hạn chế (< 16 GB)
- Chỉ cần inference, không cần full fine-tuning
- Chấp nhận giảm nhẹ chất lượng để tiết kiệm memory
- Muốn chạy nhiều model cùng lúc

**Không nên sử dụng khi**:
- Có GPU mạnh (A100, H100 với 40+ GB VRAM)
- Cần chất lượng tối đa
- Cần full fine-tuning (nhưng vẫn có thể dùng PEFT)

### Kết hợp với PEFT

**LoRA + 4-bit Quantization**:
- Load model với 4-bit quantization
- Fine-tune với LoRA (chỉ train adapter weights)
- Kết hợp tốt nhất của cả hai: tiết kiệm memory + fine-tune hiệu quả

**Ví dụ**:
```python
# Load model với quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)

# Fine-tune với LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)
```

### Tài liệu tham khảo

- [Qwen Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [BitsAndBytes Quantization](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)
- [4-bit Quantization Paper](https://arxiv.org/abs/2305.14314)