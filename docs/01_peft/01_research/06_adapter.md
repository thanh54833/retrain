# Adapters: Các Phương Pháp Adapter-Based trong PEFT

## Giới Thiệu

Các phương pháp **adapter-based** thêm các tham số có thể huấn luyện bổ sung sau các lớp attention và fully-connected của một model pretrained đã được đóng băng (frozen) để giảm sử dụng bộ nhớ và tăng tốc huấn luyện. Phương pháp khác nhau tùy thuộc vào adapter, có thể đơn giản là một lớp bổ sung hoặc có thể biểu diễn các cập nhật trọng số ∆W như một phân rã hạng thấp của ma trận trọng số. Dù bằng cách nào, các adapter thường nhỏ nhưng thể hiện hiệu suất tương đương với một model được fine-tune đầy đủ và cho phép huấn luyện các model lớn hơn với ít tài nguyên hơn.

Hướng dẫn này sẽ cung cấp cho bạn tổng quan ngắn gọn về các phương pháp adapter được hỗ trợ bởi PEFT.

## 1. Low-Rank Adaptation (LoRA)

> [!TIP]
> LoRA là một trong những phương pháp PEFT phổ biến nhất và là điểm khởi đầu tốt nếu bạn mới bắt đầu với PEFT. Nó ban đầu được phát triển cho các large language models nhưng là một phương pháp huấn luyện cực kỳ phổ biến cho diffusion models do tính hiệu quả và hiệu quả của nó.

[LoRA](https://hf.co/papers/2106.09685) là một kỹ thuật tăng tốc fine-tuning các model lớn trong khi tiêu thụ ít bộ nhớ hơn.

### Cách Hoạt Động

LoRA biểu diễn các cập nhật trọng số ∆W bằng hai ma trận nhỏ hơn (gọi là *update matrices*) thông qua phân rã hạng thấp. Các ma trận mới này có thể được huấn luyện để thích ứng với dữ liệu mới trong khi giữ tổng số tham số ở mức thấp. Ma trận trọng số gốc vẫn được đóng băng và không nhận bất kỳ cập nhật nào. Để tạo ra kết quả cuối cùng, trọng số gốc và trọng số được điều chỉnh bổ sung được kết hợp. Bạn cũng có thể merge các trọng số adapter với model gốc để loại bỏ độ trễ inference.

### Ưu Điểm

- ✅ LoRA làm cho fine-tuning hiệu quả hơn bằng cách giảm đáng kể số lượng tham số có thể huấn luyện
- ✅ Các trọng số pretrained gốc được giữ nguyên, có nghĩa là bạn có thể có nhiều model LoRA nhẹ và di động cho các tác vụ downstream khác nhau được xây dựng trên chúng
- ✅ LoRA trực giao với các phương pháp parameter-efficient khác và có thể được kết hợp với nhiều phương pháp trong số đó
- ✅ Hiệu suất của các model được fine-tune bằng LoRA tương đương với hiệu suất của các model được fine-tune đầy đủ

### Ứng Dụng

Về nguyên tắc, LoRA có thể được áp dụng cho bất kỳ tập con nào của các ma trận trọng số trong một mạng neural để giảm số lượng tham số có thể huấn luyện. Tuy nhiên, để đơn giản và hiệu quả tham số hơn nữa, LoRA thường chỉ được áp dụng cho các attention blocks trong các model Transformer. Số lượng tham số có thể huấn luyện trong model LoRA phụ thuộc vào kích thước của các ma trận cập nhật, được xác định chủ yếu bởi rank `r` và hình dạng của ma trận trọng số gốc.

## 2. Mixture of LoRA Experts (X-LoRA)

[X-LoRA](https://huggingface.co/papers/2402.07148) là một phương pháp mixture of experts cho LoRA hoạt động bằng cách sử dụng gating dày đặc hoặc thưa thớt để kích hoạt động các LoRA experts. Các LoRA experts cũng như model gốc được đóng băng trong quá trình huấn luyện, dẫn đến số lượng tham số thấp vì chỉ các lớp gating phải được huấn luyện. Đặc biệt, các lớp gating xuất ra các scaling (tùy thuộc vào cấu hình) là chi tiết ở cấp độ layer và token.

### Đặc Điểm

- **Dual Forward Pass**: X-LoRA yêu cầu model gốc chạy hai lần cho mỗi bước: lần đầu để lấy hidden states mà không có bất kỳ LoRA adapter nào, và lần thứ hai, các hidden states được sử dụng để tính toán scalings được áp dụng cho các LoRA adapters và model chạy lần thứ hai
- **Dynamic Activation**: Trong quá trình inference, X-LoRA kích hoạt động các LoRA adapters để nhớ lại kiến thức và trộn chúng một cách hiệu quả
- **Token-by-Token Scaling**: Scalings thay đổi cho các prompt khác nhau cho mỗi token, làm nổi bật việc kích hoạt các adapter khác nhau khi generation tiến triển

### Lợi Ích

- ✅ Cho phép model phản ánh kiến thức của nó nhờ vào dual forward pass scheme
- ✅ Cấu hình lại kiến trúc động
- ✅ Hiệu quả về tham số vì chỉ gating layers được huấn luyện

## 3. Low-Rank Hadamard Product (LoHa)

Phân rã hạng thấp có thể ảnh hưởng đến hiệu suất vì các cập nhật trọng số bị giới hạn trong không gian hạng thấp, có thể hạn chế khả năng biểu đạt của model. Tuy nhiên, bạn không nhất thiết muốn sử dụng rank lớn hơn vì nó tăng số lượng tham số có thể huấn luyện. Để giải quyết vấn đề này, [LoHa](https://huggingface.co/papers/2108.06098) (một phương pháp ban đầu được phát triển cho computer vision) đã được áp dụng cho diffusion models nơi khả năng tạo ra các hình ảnh đa dạng là một cân nhắc quan trọng.

### Cách Hoạt Động

LoHa sử dụng [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (tích phần tử) thay vì tích ma trận. ∆W được biểu diễn bởi bốn ma trận nhỏ hơn thay vì hai - như trong LoRA - và mỗi cặp các ma trận hạng thấp này được kết hợp với Hadamard product. Kết quả là, ∆W có thể có cùng số lượng tham số có thể huấn luyện nhưng rank và khả năng biểu đạt cao hơn.

### Ưu Điểm

- ✅ Cùng số tham số nhưng rank cao hơn LoRA
- ✅ Khả năng biểu đạt tốt hơn
- ✅ Phù hợp cho diffusion models

## 4. Low-Rank Kronecker Product (LoKr)

[LoKr](https://hf.co/papers/2309.14859) rất giống với LoRA và LoHa, và nó cũng chủ yếu được áp dụng cho diffusion models, mặc dù bạn cũng có thể sử dụng nó với các loại model khác.

### Cách Hoạt Động

LoKr thay thế tích ma trận bằng [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) thay vì. Phân rã Kronecker product tạo ra một ma trận khối bảo toàn rank của ma trận trọng số gốc. Một lợi ích khác của Kronecker product là nó có thể được vector hóa bằng cách xếp chồng các cột ma trận. Điều này có thể tăng tốc quá trình vì bạn tránh việc tái tạo hoàn toàn ∆W.

### Ưu Điểm

- ✅ Bảo toàn rank của ma trận gốc
- ✅ Có thể vector hóa để tăng tốc
- ✅ Tránh tái tạo hoàn toàn ∆W

## 5. Orthogonal Finetuning (OFT)

[OFT](https://hf.co/papers/2306.07280) là một phương pháp chủ yếu tập trung vào việc bảo toàn hiệu suất generative của model pretrained trong model được fine-tune. Nó cố gắng duy trì cùng cosine similarity (hyperspherical energy) giữa tất cả các cặp neurons trong một lớp vì điều này nắm bắt tốt hơn thông tin ngữ nghĩa giữa các neurons.

### Cách Hoạt Động

OFT bảo toàn hyperspherical energy bằng cách học một phép biến đổi trực giao cho neurons để giữ cosine similarity giữa chúng không đổi. Trong thực tế, điều này có nghĩa là lấy tích ma trận của một ma trận trực giao với ma trận trọng số pretrained. Tuy nhiên, để parameter-efficient, ma trận trực giao được biểu diễn như một ma trận block-diagonal với các khối rank `r`.

### Đặc Điểm

- ✅ Bảo toàn subject tốt hơn
- ✅ Tốt cho controllable generation (tương tự như ControlNet)
- ✅ Giảm tham số với cấu trúc ma trận block-diagonal thưa thớt

### So Sánh với LoRA

Trong khi LoRA giảm số lượng tham số có thể huấn luyện với cấu trúc hạng thấp, OFT giảm số lượng tham số có thể huấn luyện với cấu trúc ma trận block-diagonal thưa thớt.

## 6. Orthogonal Butterfly (BOFT)

[BOFT](https://hf.co/papers/2311.06243) là một phương pháp orthogonal finetuning được cải thiện tập trung vào việc bảo toàn khả năng generative của model pretrained trong khi hiệu quả về tham số hơn đáng kể so với OFT tiêu chuẩn.

### Cách Hoạt Động

Giống như OFT, BOFT duy trì cùng cosine similarity (hyperspherical energy) giữa tất cả các cặp neurons trong một lớp bằng cách áp dụng một phép biến đổi trực giao cho ma trận trọng số pretrained. Thay vì sử dụng ma trận trực giao block-diagonal, BOFT phân tích phép biến đổi trực giao thành tích của **sparse butterfly matrices** (ban đầu được giới thiệu trong [Cooley–Tukey FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)).

### Đặc Điểm

- **Dense Connectivity**: Không giống như các phép quay block-diagonal của OFT, chỉ trộn inputs trong mỗi khối, cấu trúc butterfly đảm bảo rằng mọi input có thể ảnh hưởng đến mọi output, tạo ra **dense connectivity** với chỉ `O(d log d)` tham số
- **Preserved Expressivity**: Phân tích này bảo toàn khả năng biểu đạt trong khi giảm đáng kể số lượng tham số so với OFT (đổi lấy thời gian tính toán)

### Ưu Điểm

- ✅ Hiệu quả về tham số hơn OFT
- ✅ Bảo toàn khả năng biểu đạt
- ✅ Phù hợp cho controllable generation
- ✅ Scale tốt cho các model lớn hơn với overhead bộ nhớ và tính toán thấp hơn

## 7. Adaptive Low-Rank Adaptation (AdaLoRA)

[AdaLoRA](https://hf.co/papers/2303.10512) quản lý ngân sách tham số được giới thiệu từ LoRA bằng cách phân bổ nhiều tham số hơn - nói cách khác, rank `r` cao hơn - cho các ma trận trọng số quan trọng được thích ứng tốt hơn cho một tác vụ và cắt tỉa những ma trận ít quan trọng hơn.

### Cách Hoạt Động

Rank được kiểm soát bởi một phương pháp tương tự như singular value decomposition (SVD). ∆W được tham số hóa với hai ma trận trực giao và một ma trận đường chéo chứa các singular values. Phương pháp tham số hóa này tránh việc áp dụng SVD lặp đi lặp lại, điều này tốn kém về mặt tính toán. Dựa trên phương pháp này, rank của ∆W được điều chỉnh theo một điểm quan trọng. ∆W được chia thành các bộ ba và mỗi bộ ba được chấm điểm theo đóng góp của nó cho hiệu suất model. Các bộ ba có điểm quan trọng thấp bị cắt tỉa và các bộ ba có điểm quan trọng cao được giữ lại để fine-tuning.

### Các Giai Đoạn Huấn Luyện

Huấn luyện với AdaLoRA có ba giai đoạn:

1. **Init Phase**: Không áp dụng budgeting, do đó ranks không bị chạm vào
2. **Budgeting Phase**: Quá trình mô tả ở trên được áp dụng và rank được phân phối lại theo ngân sách, nhằm mục đích cung cấp nhiều rank hơn cho các adapter quan trọng hơn và ít rank hơn cho các lớp ít quan trọng hơn
3. **Final Phase**: Budgeting đã kết thúc, ranks được phân phối lại nhưng chúng ta có thể tiếp tục huấn luyện một thời gian với các ranks đã phân phối lại để cải thiện hiệu suất hơn nữa

### Ưu Điểm

- ✅ Phân bổ tham số thông minh
- ✅ Tự động điều chỉnh rank
- ✅ Hiệu quả hơn LoRA cho một số tác vụ

## 8. Llama-Adapter

[Llama-Adapter](https://hf.co/papers/2303.16199) là một phương pháp để thích ứng Llama thành một model tuân theo hướng dẫn. Để giúp thích ứng model cho instruction-following, adapter được huấn luyện với một dataset instruction-output 52K.

### Cách Hoạt Động

Một tập hợp các adaptation prompts có thể học được được prefix vào các instruction tokens đầu vào. Chúng được chèn vào các lớp trên của model vì tốt hơn là học với ngữ nghĩa cấp cao của model pretrained. Các instruction-output tokens được prefix vào đầu vào hướng dẫn adaptation prompt để tạo ra một phản hồi ngữ cảnh.

### Đặc Điểm

- **Zero-Initialized Attention**: Để tránh thêm noise vào tokens, adapter sử dụng attention khởi tạo bằng không
- **Learnable Gating Factor**: Adapter thêm một hệ số gating có thể học được (khởi tạo bằng không) để dần dần thêm thông tin vào model trong quá trình huấn luyện
- **Progressive Information Addition**: Điều này ngăn chặn việc làm quá tải kiến thức pretrained của model với các hướng dẫn mới được học

### Ưu Điểm

- ✅ Hiệu quả cho instruction-following
- ✅ Zero-init attention tránh noise
- ✅ Gating factor cho phép thêm thông tin dần dần

## 9. Householder Reflection Adaptation (HRA)

[HRA](https://huggingface.co/papers/2405.17484) cung cấp một góc nhìn mới kết nối LoRA với OFT, có nghĩa là nó có thể khai thác ưu điểm của cả hai chiến lược, giảm tham số và chi phí tính toán trong khi phạt mất kiến thức pre-training.

### Cách Hoạt Động

HRA xây dựng một chuỗi `r` Householder reflections (HRs) có thể huấn luyện. Vì ma trận Householder reflection là một ma trận trực giao và tích của các ma trận trực giao cũng là một ma trận trực giao, HRA thỏa mãn đảm bảo lý thuyết của Orthogonal Finetuning (OFT). Đồng thời, HRA cũng có thể được xem như một adapter fine-tuning hạng thấp bằng cách viết lại công thức.

### Đặc Điểm

- **Rank Control**: Rank `r` càng cao, càng nhiều tham số có thể huấn luyện, dẫn đến khả năng model lớn hơn và hiệu suất tốt hơn
- **Orthogonality Regularizer**: Do cấu trúc chuỗi, tính trực giao của các mặt phẳng HR ảnh hưởng đến khả năng và tính quy tắc của HRA. Để đạt được sự cân bằng giữa khả năng model và tính quy tắc, một regularizer trực giao của các mặt phẳng HR được thêm vào hàm loss. Trọng số λ có thể kiểm soát độ mạnh của regularizer

### Ưu Điểm

- ✅ Kết hợp ưu điểm của LoRA và OFT
- ✅ Giảm tham số và chi phí tính toán
- ✅ Bảo toàn kiến thức pre-training tốt hơn

## 10. MiSS (Matrix Shard Sharing)

[MiSS](https://huggingface.co/papers/2409.15371) (Matrix Shard Sharing) là một phương pháp Parameter-Efficient Fine-Tuning (PEFT) mới được thiết kế để giải quyết sự cân bằng giữa khả năng thích ứng và hiệu quả trong Large Language Models.

### Cách Hoạt Động

Cách tiếp cận cốt lõi của MiSS liên quan đến một cơ chế shard-sharing đơn giản. Nó đạt được low-rank adaptation bằng cách phân tích một ma trận trọng số thành nhiều fragments và sau đó sử dụng một "common fragment" có thể huấn luyện được chia sẻ. Ma trận cập nhật hạng thấp cuối cùng được xây dựng bằng cách sao chép các shards được chia sẻ, được phân vùng này.

### Đặc Điểm

- **Single Trainable Matrix**: MiSS chỉ yêu cầu một ma trận có thể huấn luyện duy nhất
- **Shape Consistency**: Hình dạng của một ma trận có thể huấn luyện duy nhất trong MiSS nhất quán với `lora_B`, vì vậy tham số `r` trong MiSS nhỏ hơn `r` trong LoRA bởi (`in_feature * r`)
- **Special Requirements**: Bat's r (b) đặc biệt và yêu cầu trọng số W thỏa mãn các điều kiện `in_features % r == 0` và `out_features % r == 0`
- **Parameter Efficiency**: Khi `in_features == out_features` và MiSS-r bằng LoRA-r, số lượng tham số có thể huấn luyện của MiSS chỉ bằng một nửa so với LoRA

### Lưu Ý

Mặc dù các cập nhật phi tuyến của Bat mang lại một số cải thiện hiệu suất, chúng cũng tăng overhead tính toán. Mục đích chính của nó là cung cấp cho các nhà nghiên cứu một hướng cải thiện. Do đó, chúng tôi khuyến nghị fine-tuning model MiSS toàn diện thay thế.

### Ưu Điểm

- ✅ Cân bằng tốt giữa hiệu suất và hiệu quả
- ✅ Ít tham số hơn LoRA trong một số trường hợp
- ✅ Cơ chế shard-sharing đơn giản

## So Sánh Tổng Quan

| Phương Pháp | Số Tham Số | Hiệu Suất | Độ Phức Tạp | Use Case |
|-------------|------------|-----------|-------------|----------|
| **LoRA** | Thấp | Tốt | Thấp | Phổ biến nhất, điểm khởi đầu tốt |
| **X-LoRA** | Rất thấp | Tốt | Trung bình | Mixture of experts, dynamic activation |
| **LoHa** | Thấp | Tốt | Trung bình | Diffusion models, cần rank cao hơn |
| **LoKr** | Thấp | Tốt | Trung bình | Diffusion models, vectorization |
| **OFT** | Thấp | Rất tốt | Trung bình | Controllable generation, bảo toàn subject |
| **BOFT** | Rất thấp | Rất tốt | Cao | Controllable generation, hiệu quả hơn OFT |
| **AdaLoRA** | Thấp-Trung bình | Rất tốt | Cao | Tự động điều chỉnh rank, phân bổ thông minh |
| **Llama-Adapter** | Thấp | Tốt | Thấp | Instruction-following, Llama models |
| **HRA** | Thấp | Tốt | Trung bình | Kết hợp LoRA và OFT |
| **MiSS** | Rất thấp | Tốt | Thấp | Cân bằng hiệu suất và hiệu quả |

## Khi Nào Sử Dụng Phương Pháp Nào?

### Chọn LoRA Nếu:
- Bạn mới bắt đầu với PEFT
- Cần giải pháp phổ biến và được hỗ trợ tốt
- Làm việc với LLM hoặc diffusion models

### Chọn X-LoRA Nếu:
- Cần mixture of experts
- Muốn dynamic activation của adapters
- Có nhiều LoRA adapters cần quản lý

### Chọn LoHa/LoKr Nếu:
- Làm việc với diffusion models
- Cần rank cao hơn nhưng không muốn tăng tham số
- Cần vectorization để tăng tốc

### Chọn OFT/BOFT Nếu:
- Cần controllable generation
- Cần bảo toàn subject tốt
- Làm việc với text-to-image generation

### Chọn AdaLoRA Nếu:
- Cần tự động điều chỉnh rank
- Muốn phân bổ tham số thông minh
- Có ngân sách tham số cố định

### Chọn Llama-Adapter Nếu:
- Làm việc với Llama models
- Cần instruction-following
- Muốn giải pháp đơn giản

### Chọn HRA Nếu:
- Muốn kết hợp ưu điểm của LoRA và OFT
- Cần bảo toàn kiến thức pre-training tốt hơn

### Chọn MiSS Nếu:
- Cần cân bằng tốt giữa hiệu suất và hiệu quả
- Muốn ít tham số hơn LoRA trong một số trường hợp

## Kết Luận

Các phương pháp adapter-based trong PEFT cung cấp nhiều cách tiếp cận khác nhau để fine-tune các model lớn một cách hiệu quả. Mỗi phương pháp có ưu và nhược điểm riêng, phù hợp với các use case khác nhau:

- **LoRA** vẫn là điểm khởi đầu tốt nhất cho hầu hết các trường hợp
- **OFT/BOFT** tốt cho controllable generation
- **AdaLoRA** tốt khi cần tự động điều chỉnh
- **X-LoRA** tốt cho mixture of experts
- **MiSS** tốt cho cân bằng hiệu suất và hiệu quả

Việc lựa chọn phương pháp phụ thuộc vào tài nguyên có sẵn, yêu cầu hiệu suất, và đặc thù của tác vụ.

## Tài Liệu Tham Khảo

- [Hugging Face PEFT - Adapters Conceptual Guide](https://huggingface.co/docs/peft/en/conceptual_guides/adapter)
- [LoRA Paper](https://hf.co/papers/2106.09685)
- [X-LoRA Paper](https://huggingface.co/papers/2402.07148)
- [LoHa Paper](https://huggingface.co/papers/2108.06098)
- [LoKr Paper](https://hf.co/papers/2309.14859)
- [OFT Paper](https://hf.co/papers/2306.07280)
- [BOFT Paper](https://hf.co/papers/2311.06243)
- [AdaLoRA Paper](https://hf.co/papers/2303.10512)
- [Llama-Adapter Paper](https://hf.co/papers/2303.16199)
- [HRA Paper](https://huggingface.co/papers/2405.17484)
- [MiSS Paper](https://huggingface.co/papers/2409.15371)

