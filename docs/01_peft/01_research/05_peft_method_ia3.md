# IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations

## Giới Thiệu

**IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)** nhân các activations của model (keys và values trong self-attention và encoder-decoder attention blocks, và intermediate activation của position-wise feedforward network) với ba vector đã học được. Phương pháp PEFT này giới thiệu số lượng tham số có thể huấn luyện thậm chí còn nhỏ hơn LoRA, vì LoRA giới thiệu các ma trận trọng số thay vì vector. Các tham số của model gốc được giữ nguyên (frozen) và chỉ các vector này được cập nhật. Kết quả là nó nhanh hơn, rẻ hơn và hiệu quả hơn để fine-tune cho một tác vụ downstream mới.

> [!TIP]
> Một số hiểu biết về quá trình huấn luyện sequence-to-sequence nói chung sẽ rất hữu ích và cho phép bạn tập trung vào cách áp dụng IA3. Nếu bạn mới bắt đầu, chúng tôi khuyến nghị xem các hướng dẫn [Translation](https://huggingface.co/docs/transformers/tasks/translation) và [Summarization](https://huggingface.co/docs/transformers/tasks/summarization) trước từ tài liệu Transformers. Khi bạn đã sẵn sàng, quay lại và xem cách dễ dàng áp dụng PEFT vào quá trình huấn luyện của bạn!

## Cách IA3 Hoạt Động

IA3 nhân các activations của model với ba vector đã học được:

1. **Keys trong attention blocks**: Vector cho keys trong self-attention và encoder-decoder attention
2. **Values trong attention blocks**: Vector cho values trong self-attention và encoder-decoder attention  
3. **Intermediate activations**: Vector cho intermediate activation của position-wise feedforward network

Điều này cho phép model điều chỉnh cách xử lý thông tin mà không cần cập nhật toàn bộ trọng số.

## Dataset

Trong hướng dẫn này, chúng ta sẽ sử dụng tập con `sentences_allagree` của dataset [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank). Tập con này chứa tin tức tài chính với 100% sự đồng ý của người chú thích về nhãn sentiment.

### Tải Dataset

```python
from datasets import load_dataset

ds = load_dataset("financial_phrasebank", "sentences_allagree")
ds = ds["train"].train_test_split(test_size=0.1)
ds["validation"] = ds["test"]
del ds["test"]

# Tạo cột text_label để dễ hiểu hơn
classes = ds["train"].features["label"].names
ds = ds.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

# Xem ví dụ
ds["train"][0]
# {'sentence': 'It will be operated by Nokia , and supported by its Nokia NetAct network and service management system .',
#  'label': 1,
#  'text_label': 'neutral'}
```

### Chuẩn Bị Tokenizer

```python
from transformers import AutoTokenizer

text_column = "sentence"
label_column = "text_label"
max_length = 128

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")
```

### Hàm Tiền Xử Lý

Tạo hàm tiền xử lý để:
1. Tokenize inputs, padding và truncate sequence đến `max_length`
2. Áp dụng cùng tokenizer cho labels nhưng với `max_length` ngắn hơn tương ứng với label
3. Mask các padding tokens

```python
def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Tokenize labels với max_length ngắn hơn
    labels = tokenizer(
        targets, 
        max_length=3, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    labels = labels["input_ids"]
    
    # Mask padding tokens
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs
```

### Áp Dụng Tiền Xử Lý

```python
processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```

### Tạo DataLoader

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_ds = processed_ds["train"]
eval_ds = processed_ds["validation"]

batch_size = 8

train_dataloader = DataLoader(
    train_ds, 
    shuffle=True, 
    collate_fn=default_data_collator, 
    batch_size=batch_size, 
    pin_memory=True
)
eval_dataloader = DataLoader(
    eval_ds, 
    collate_fn=default_data_collator, 
    batch_size=batch_size, 
    pin_memory=True
)
```

## Model

Tải một model pretrained để sử dụng làm base model cho IA3. Hướng dẫn này sử dụng model [bigscience/mt0-large](https://huggingface.co/bigscience/mt0-large), nhưng bạn có thể sử dụng bất kỳ sequence-to-sequence model nào bạn thích.

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
```

### Cấu Hình IA3 và Model

Tất cả các phương pháp PEFT cần một cấu hình chứa và chỉ định tất cả các tham số cho cách phương pháp PEFT nên được áp dụng. Tạo một [`IA3Config`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/ia3#peft.IA3Config) với task type và đặt inference mode thành `False`.

> [!TIP]
> Gọi phương thức [`print_trainable_parameters()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel.print_trainable_parameters) để so sánh số lượng tham số có thể huấn luyện của [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) so với số lượng tham số trong model gốc!

```python
from peft import IA3Config, get_peft_model

peft_config = IA3Config(task_type="SEQ_2_SEQ_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: "trainable params: 282,624 || all params: 1,229,863,936 || trainable%: 0.022980103060766553"
```

### Các Tham Số IA3Config

Bạn có thể tìm thêm các tham số cho cấu hình này trong [API reference](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/ia3#ia3config):

- `task_type`: Loại tác vụ (SEQ_2_SEQ_LM, CAUSAL_LM, etc.)
- `inference_mode`: Chế độ inference (True/False)
- `target_modules`: Các module đích để áp dụng IA3
- `feedforward_modules`: Các module feedforward để áp dụng IA3

## Huấn Luyện

### Thiết Lập Optimizer và Scheduler

```python
import torch
from transformers import get_linear_schedule_with_warmup

lr = 8e-3
num_epochs = 3

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

### Vòng Lặp Huấn Luyện

Di chuyển model đến accelerator và tạo vòng lặp huấn luyện báo cáo loss và perplexity cho mỗi epoch:

```python
from tqdm import tqdm

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model = model.to(device)

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Evaluation
    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(
                torch.argmax(outputs.logits, -1).detach().cpu().numpy(), 
                skip_special_tokens=True
            )
        )

    # Tính toán metrics
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

## Lưu và Chia Sẻ Model

Sau khi huấn luyện hoàn tất, bạn có thể tải model lên Hub bằng phương thức [`push_to_hub`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub). Bạn sẽ cần đăng nhập vào tài khoản Hugging Face của mình trước và nhập token khi được nhắc.

```python
from huggingface_hub import notebook_login

# Đăng nhập vào Hugging Face
notebook_login()

# Tải model lên Hub
account = "your-username"
peft_model_id = f"{account}/mt0-large-ia3"
model.push_to_hub(peft_model_id)
```

## Inference

Để tải model cho inference, sử dụng phương thức [`from_pretrained()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/auto_class#peft.AutoPeftModel.from_pretrained). Hãy cũng tải một câu tin tức tài chính từ dataset để generate sentiment:

```python
from peft import AutoPeftModelForSeq2SeqLM

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

# Tải model đã huấn luyện
model = AutoPeftModelForSeq2SeqLM.from_pretrained("your-username/mt0-large-ia3").to(device)
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

# Chuẩn bị input
i = 15
inputs = tokenizer(ds["validation"][text_column][i], return_tensors="pt")
print(ds["validation"][text_column][i])
# "The robust growth was the result of the inclusion of clothing chain Lindex in the Group in December 2007 ."
```

Gọi phương thức [`generate`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) để generate nhãn sentiment dự đoán:

```python
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
# ['positive']
```

## So Sánh IA3 với LoRA

| Tiêu Chí | IA3 | LoRA |
|----------|-----|------|
| **Số tham số trainable** | Rất ít (vector) | Ít (ma trận) |
| **Cách hoạt động** | Nhân activations với vector | Thêm ma trận hạng thấp |
| **Hiệu suất** | Tốt | Tốt |
| **Tốc độ huấn luyện** | Nhanh hơn | Nhanh |
| **Yêu cầu bộ nhớ** | Rất thấp | Thấp |
| **Phù hợp cho** | Sequence-to-sequence, Causal LM | Nhiều loại model |

## Ưu và Nhược Điểm

### Ưu Điểm

- ✅ **Rất ít tham số**: Ít tham số trainable hơn cả LoRA
- ✅ **Nhanh**: Huấn luyện nhanh hơn do ít tham số hơn
- ✅ **Hiệu quả**: Tiết kiệm bộ nhớ và tài nguyên tính toán
- ✅ **Đơn giản**: Chỉ cần nhân activations với vector
- ✅ **Phù hợp cho Seq2Seq**: Hoạt động tốt với sequence-to-sequence models

### Nhược Điểm

- ❌ **Ít linh hoạt hơn LoRA**: Không thể điều chỉnh rank như LoRA
- ❌ **Phụ thuộc vào activations**: Hiệu suất phụ thuộc vào cách activations được xử lý
- ❌ **Ít tài liệu hơn**: So với LoRA, có ít tài liệu và ví dụ hơn

## Best Practices

### 1. Learning Rate

- IA3 thường cần learning rate cao hơn (5e-3 đến 1e-2)
- Thử nghiệm với các learning rate khác nhau
- Sử dụng learning rate scheduler

### 2. Epochs

- IA3 thường hội tụ nhanh hơn LoRA
- Bắt đầu với 3-5 epochs
- Monitor validation loss để tránh overfitting

### 3. Task Type

- Đảm bảo chọn đúng `task_type` (SEQ_2_SEQ_LM, CAUSAL_LM, etc.)
- Task type ảnh hưởng đến cách IA3 được áp dụng

### 4. Target Modules

- Có thể chỉ định `target_modules` nếu cần
- Mặc định, IA3 sẽ tự động chọn các module phù hợp

### 5. Inference Mode

- Đặt `inference_mode=False` khi huấn luyện
- Đặt `inference_mode=True` khi inference (mặc định)

## Use Cases

IA3 đặc biệt phù hợp cho:

1. **Sequence-to-Sequence Tasks**:
   - Translation
   - Summarization
   - Text generation

2. **Sentiment Analysis**:
   - Phân loại sentiment
   - Phân tích cảm xúc

3. **Khi cần ít tham số nhất**:
   - Tài nguyên rất hạn chế
   - Cần nhiều adapter cho nhiều tác vụ

4. **Prototype nhanh**:
   - Thử nghiệm nhanh với ít tài nguyên
   - Đánh giá hiệu suất ban đầu

## Kết Luận

IA3 là một phương pháp PEFT hiệu quả cao, đặc biệt phù hợp cho các tác vụ sequence-to-sequence. Với số lượng tham số trainable rất ít (thậm chí ít hơn LoRA), IA3 cung cấp một cách nhanh chóng và tiết kiệm để fine-tune các model lớn cho các tác vụ downstream mới.

**Khi nào sử dụng IA3**:
- Khi bạn cần ít tham số trainable nhất có thể
- Khi làm việc với sequence-to-sequence models
- Khi tài nguyên rất hạn chế
- Khi cần prototype nhanh

**Khi nào không sử dụng IA3**:
- Khi bạn cần điều chỉnh rank (sử dụng LoRA thay thế)
- Khi cần nhiều tài liệu và ví dụ (LoRA có nhiều hơn)
- Khi làm việc với các model không phải seq2seq (có thể LoRA tốt hơn)

## Tài Liệu Tham Khảo

- [Hugging Face PEFT - IA3 Guide](https://huggingface.co/docs/peft/en/task_guides/ia3)
- [IA3 Conceptual Guide](https://huggingface.co/docs/peft/en/conceptual_guides/ia3)
- [IA3Config API Reference](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/ia3#ia3config)
- [Financial Phrasebank Dataset](https://huggingface.co/datasets/financial_phrasebank)

