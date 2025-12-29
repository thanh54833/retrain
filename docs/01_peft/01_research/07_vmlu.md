# VMLU: Vietnamese Multitask Language Understanding Benchmark

## Giới Thiệu

**VMLU (Vietnamese Multitask Language Understanding)** là một bộ benchmark tập trung vào con người, được thiết kế đặc biệt để đánh giá khả năng tổng thể của các foundation models, với sự chuyên biệt mạnh mẽ cho tiếng Việt. Benchmark này bao gồm bốn dataset riêng biệt: **Vi-MQA**, **Vi-SQuAD**, **Vi-DROP**, và **Vi-Dialog** — mỗi dataset nhắm đến một khía cạnh khác nhau về hiệu suất của LLM, bao gồm kiến thức tổng quát, đọc hiểu, lý luận logic, và khả năng hội thoại.

Bằng cách cung cấp các tác vụ đánh giá toàn diện và đa dạng, VMLU giúp làm phong phú các benchmark đánh giá NLP tiếng Việt, thúc đẩy sự phát triển của các foundation models mạnh mẽ hơn và khuyến khích nghiên cứu thêm trong lĩnh vực LLM.

## Tổng Quan

VMLU được phát triển bởi **ZaloAI** và **JAIST** (Japan Advanced Institute of Science and Technology) vào năm 2025. Đây là một benchmark quan trọng cho việc đánh giá các mô hình ngôn ngữ lớn trong tiếng Việt.

### Mục Tiêu

- Đánh giá khả năng tổng thể của foundation models
- Tập trung đặc biệt vào tiếng Việt
- Cung cấp đánh giá đa dạng về nhiều khía cạnh của LLM
- Thúc đẩy nghiên cứu và phát triển trong lĩnh vực NLP tiếng Việt

## Các Dataset Trong VMLU

### 1. Vi-MQA (Vietnamese Multiple-choice Question Answering)

**Vi-MQA** là một benchmark trả lời câu hỏi trắc nghiệm được thiết kế để đánh giá kiến thức tổng quát và khả năng lý luận. Nó bao gồm các câu hỏi trải dài từ mức độ cơ bản đến chuyên môn cao cấp.

#### Đặc Điểm

- **58 môn học khác nhau**, với đa số chứa khoảng 200 câu hỏi mỗi môn
- **4 lĩnh vực chính**: STEM, Humanities, Social Sciences, và Others
- **4 mức độ khó**: Elementary School, Middle High School, High School, và Professional level

#### Nguồn Dữ Liệu

Dataset chủ yếu đến từ:
- Các kỳ thi của các cơ sở giáo dục uy tín (tiểu học, trung học cơ sở, trung học phổ thông, đại học)
- Kỳ thi tốt nghiệp trung học phổ thông do Bộ Giáo dục và Đào tạo tổ chức

#### Phân Loại Môn Học

##### STEM (Khoa Học, Công Nghệ, Kỹ Thuật, Toán Học)

1. Elementary Mathematics (Toán Tiểu Học)
2. Elementary Science (Khoa Học Tiểu Học)
3. Middle School Biology (Sinh Học THCS)
4. Middle School Chemistry (Hóa Học THCS)
5. Middle School Mathematics (Toán THCS)
6. Middle School Physics (Vật Lý THCS)
7. High School Biology (Sinh Học THPT)
8. High School Chemistry (Hóa Học THPT)
9. High School Mathematics (Toán THPT)
10. High School Physics (Vật Lý THPT)
11. Applied Informatics (Tin Học Ứng Dụng)
12. Computer Architecture (Kiến Trúc Máy Tính)
13. Computer Network (Mạng Máy Tính)
14. Discrete Mathematics (Toán Rời Rạc)
15. Electrical Engineering (Kỹ Thuật Điện)
16. Introduction to Chemistry (Hóa Học Đại Cương)
17. Introduction to Physics (Vật Lý Đại Cương)
18. Introduction to Programming (Lập Trình Cơ Bản)
19. Metrology Engineer (Kỹ Sư Đo Lường)
20. Operating System (Hệ Điều Hành)
21. Statistics and Probability (Thống Kê và Xác Suất)

##### Social Sciences (Khoa Học Xã Hội)

22. Middle School Civil Education (Giáo Dục Công Dân THCS)
23. Middle School Geography (Địa Lý THCS)
24. High School Civil Education (Giáo Dục Công Dân THPT)
25. High School Geography (Địa Lý THPT)
26. Business Administration (Quản Trị Kinh Doanh)
27. Ho Chi Minh Ideology (Tư Tưởng Hồ Chí Minh)
28. Macroeconomics (Kinh Tế Vĩ Mô)
29. Microeconomics (Kinh Tế Vi Mô)
30. Principles of Marxism and Leninism (Nguyên Lý Chủ Nghĩa Mác-Lênin)
31. Sociology (Xã Hội Học)

##### Humanities (Nhân Văn)

32. Elementary History (Lịch Sử Tiểu Học)
33. Middle School History (Lịch Sử THCS)
34. Middle School Literature (Ngữ Văn THCS)
35. High School History (Lịch Sử THPT)
36. High School Literature (Ngữ Văn THPT)
37. Administrative Law (Luật Hành Chính)
38. Business Law (Luật Kinh Doanh)
39. Civil Law (Luật Dân Sự)
40. Criminal Law (Luật Hình Sự)
41. Economic Law (Luật Kinh Tế)
42. Education Law (Luật Giáo Dục)
43. History of World Civilization (Lịch Sử Văn Minh Thế Giới)
44. Idealogical and Moral Cultivation (Giáo Dục Công Dân và Đạo Đức)
45. Introduction to Laws (Luật Học Đại Cương)
46. Introduction to Vietnam Culture (Văn Hóa Việt Nam Đại Cương)
47. Logic (Logic Học)
48. Revolutionary Policy of the Vietnamese Communist Party (Đường Lối Cách Mạng của Đảng Cộng Sản Việt Nam)
49. Vietnamese Language and Literature (Tiếng Việt và Văn Học)

##### Others (Khác)

50. Accountant (Kế Toán)
51. Clinical Pharmacology (Dược Lý Lâm Sàng)
52. Environmental Engineering (Kỹ Thuật Môi Trường)
53. Internal Basic Medicine (Y Học Cơ Sở)
54. Preschool Pedagogy (Giáo Dục Mầm Non)
55. Tax Accountant (Kế Toán Thuế)
56. Tax Civil Servant (Công Chức Thuế)
57. Civil Servant (Công Chức)
58. Driving License Certificate (Bằng Lái Xe)

#### Ví Dụ Câu Hỏi

**Toán Tiểu Học (STEM)**
```
Tính chất nào sau đây không phải là tính chất của thủy tinh chất lượng cao:

A. Rất trong
B. Bền, khó vỡ
C. Chịu được nóng, lạnh
D. Dễ cháy

Đáp án: D
```

**Địa Lý THCS (Social Science)**
```
Việc phát triển nông-lâm-thủy sản tạo cơ sở nguyên liệu cho ngành phát triển công nghiệp nào?

A. Công nghiệp năng lượng
B. Công nghiệp chế biến lương thực thực phẩm
C. Công nghiệp hóa chất
D. Công nghiệp sản xuất vật liệu xây dựng

Đáp án: B
```

**Lịch Sử THPT (Humanity)**
```
Sự kiện nào sau đây đã tạo ra một cơ chế giải quyết các vấn đề liên quan đến hòa bình và an ninh ở châu Âu?

A. Định ước Henxinki (08/1975)
B. Liên Xô và Mỹ ký Hiệp định hạn chế vũ khí tiến công chiến lược
C. Mỹ và Liên Xô tuyên bố chấm dứt Chiến tranh lạnh
D. Hiệp định về những cơ sở của quan hệ giữa Đông Đức và Tây Đức

Đáp án: A
```

**Dược Lý Lâm Sàng (Others)**
```
Khái niệm DƯỢC LỰC HỌC:

A. Động học của sự hấp thu, phân phối, chuyển hóa và thải trừ thuốc
B. Nghiên cứu tác động của thuốc trên cơ thể sống
C. Nghiên cứu về tác động của cơ thể đến thuốc
D. Là môn khoa học nghiên cứu về thuốc

Đáp án: B
```

### 2. Vi-SQuAD (Vietnamese Stanford Question Answering Dataset)

**Vi-SQuAD v1.0** là phiên bản tiếng Việt của Stanford Question Answering Dataset, được thiết kế để đánh giá khả năng đọc hiểu của mô hình.

#### Đặc Điểm

- Đánh giá khả năng đọc hiểu văn bản
- Câu hỏi dựa trên đoạn văn cho trước
- Yêu cầu mô hình trích xuất câu trả lời từ ngữ cảnh

### 3. Vi-DROP (Vietnamese Discrete Reasoning Over Paragraphs)

**Vi-DROP v1.0** là phiên bản tiếng Việt của DROP dataset, tập trung vào lý luận rời rạc qua các đoạn văn.

#### Đặc Điểm

- Đánh giá khả năng lý luận logic
- Yêu cầu thực hiện các phép tính và suy luận
- Kiểm tra khả năng hiểu và xử lý thông tin số học

### 4. Vi-Dialog (Vietnamese Dialogue Dataset)

**Vi-Dialog v1.0** là dataset hội thoại tiếng Việt, được thiết kế để đánh giá khả năng hội thoại của mô hình.

#### Đặc Điểm

- Đánh giá khả năng hội thoại tự nhiên
- Kiểm tra khả năng duy trì ngữ cảnh trong cuộc trò chuyện
- Đánh giá tính phù hợp và tự nhiên của phản hồi

## Tải Dataset

Các dataset VMLU có thể được tải xuống như sau:

- **Vi-MQA v1.5** - Vietnamese Multiple-choice Question Answering
- **Vi-SQuAD v1.0** - Vietnamese Stanford Question Answering Dataset
- **Vi-DROP v1.0** - Vietnamese Discrete Reasoning Over Paragraphs
- **Vi-Dialog v1.0** - Vietnamese Dialogue Dataset

## GitHub Repository

Repository VMLU cung cấp thông tin chi tiết về dataset, bao gồm:

- Số lượng câu hỏi trong mỗi môn học
- Hướng dẫn sử dụng
- Code mẫu để sử dụng dataset
- Kết quả benchmark chi tiết cho các mô hình công khai
- Giải thích về các kỹ thuật prompting được sử dụng
- Metrics được sử dụng để đánh giá
- Code benchmark có thể truy cập để tạo lại kết quả

## Ứng Dụng

### Đánh Giá Mô Hình

VMLU được sử dụng để:
- Đánh giá hiệu suất của các foundation models trên tiếng Việt
- So sánh các mô hình khác nhau
- Theo dõi sự tiến bộ trong nghiên cứu NLP tiếng Việt

### Fine-tuning và Evaluation

- Sử dụng làm dataset đánh giá sau khi fine-tuning
- Đánh giá khả năng zero-shot và few-shot learning
- Kiểm tra khả năng chuyển giao kiến thức

### Nghiên Cứu

- Nghiên cứu về khả năng hiểu ngôn ngữ của LLM
- Phát triển các kỹ thuật cải thiện hiệu suất
- Phân tích điểm mạnh và điểm yếu của các mô hình

## Liên Hệ

- **Câu hỏi về VMLU**: contact@vmlu.ai hoặc tạo issue trên GitHub
- **Hợp tác tiềm năng**: developer@vmlu.ai

## Tài Liệu Tham Khảo

- [VMLU Official Website](https://vmlu.ai/)
- [VMLU GitHub Repository](https://github.com/vmlu-ai/vmlu) (tham khảo từ website)
- ZaloAI và JAIST (2025). VMLU: Vietnamese Multitask Language Understanding Benchmark

## Kết Luận

VMLU là một benchmark quan trọng và toàn diện cho việc đánh giá các mô hình ngôn ngữ lớn trong tiếng Việt. Với 4 dataset đa dạng và 58 môn học trong Vi-MQA, VMLU cung cấp một công cụ đánh giá mạnh mẽ để thúc đẩy sự phát triển của các foundation models cho tiếng Việt.

Benchmark này đặc biệt hữu ích cho:
- Các nhà nghiên cứu phát triển LLM cho tiếng Việt
- Đánh giá hiệu suất của các mô hình đã fine-tune
- So sánh và cải thiện các phương pháp PEFT trên tiếng Việt
- Nghiên cứu về khả năng hiểu ngôn ngữ của AI

