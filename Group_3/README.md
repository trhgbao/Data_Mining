# Data Mining Project: Plant Disease Classification & Preprocessing Pipeline

##  Danh sách thành viên

| STT | Họ và tên            |   MSSV   |
| --- | -------------------- | -------- |
| 1   | Bùi Anh Quân         | 23122017 |
| 2   | Trần Hoàng Gia Bảo   | 23122022 |
| 3   | Chung Tín Đạt        | 23122024 |
| 4   | Ngô Thị Thục Quyên   | 23120348 |
| 3   | Lăng Phú Quý         | 23120415 |



## Tổng quan dự án
- Tiền xử lý dữ liệu là làm sạch dữ liệu, là một quy trình kỹ thuật đòi hỏi hiểu biết sâu về thống kê,
thiết kế hệ thống và khả năng lập luận phân tích. Đồ án áp dụng các kỹ thuật tiền xử lý, khả năng giải thích
bằng các công cụ thống kê, so sánh có kiểm soát và thiết kế pipeline có thể sử dụng lại được cho các loại dữ liệu sau:
  • Dữ liệu ảnh: Biến đổi không gian, phân tích phân phối đặc trưng, augmentation có
  kiểm soát.
  • Dữ liệu bảng: Phát hiện ngoại lai, kiểm định thống kê, giảm chiều dữ liệu để trực quan
  hoá/ trích chọn đặc trưng.
  • Dữ liệu văn bản: Tiền xử lý theo đặc tính ngôn ngữ, biểu diễn ngữ nghĩa, phân tích
  tính thưa của dữ liệu.

### Xử lý văn bản dạng ảnh
- EDA và xây dựng quy trình tiền xử lý cho tập dữ liệu hình ảnh bệnh lý lá cây.
- Mục tiêu là làm sạch dữ liệu, trích xuất đặc trưng hình học/màu sắc.
### Xử lý văn bản dạng bảng
### Xử lý văn bản dạng văn bản


---

## Cấu trúc thư mục nộp bài
```bash
Group_3/
|-- README.md                                  # Tổng quan, hướng dẫn chạy, link tài nguyên
|-- requirements.txt                           # Danh sách thư viện và version cụ thể
|-- data/
|   |-- raw/                                   # Chứa dữ liệu gốc từ Kaggle
|   |   |-- lab_2.1_NewPlantDiseasesDataset    # Tập dữ liệu ảnh số
|   |   |   |-- class_folders
|   |   |   |-- images
|   |   |   |-- labels.csv
|   |   |--                                    # Tập dữ liệu dạng bảng
|   |   |--                                    # Tập dữ liệu văn bản
|   |-- processed/                             # Chứa dữ liệu sau tiền xử lý (CSV, ảnh biểu đồ, dữ liệu sau tiền xử lý)
|   |   |--  processed_image
|   |   |--  processed_tabular
|   |   |--  processed_text
|-- notebooks/
|   |-- 01_EDA_image.ipynb     
|   |-- 02_preprocessing_image.ipynb 
|   |-- 03_EDA_tabular.ipynb
|   |-- 04_preprocessing_tabular.ipynb
|   |-- 05_text_preprocessing.ipynb
|-- docs/
|   |-- Report.pdf                             # Báo cáo chi tiết kết quả thực nghiệm 
```

---

## Yêu cầu hệ thống & Cài đặt

### 1. Phiên bản Python
*   **Python:** khuyến nghị phiên bản **3.10** hoặc **3.11**.

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
*Xử lý dữ liệu ảnh:*
*   `opencv-python` (4.10.0)
*   `scikit-image` (0.24.0) - Tính SSIM/PSNR
*   `scikit-learn` (1.5.0) - PCA, IncrementalPCA, Logistic Regression
*   `ImageHash` (4.3.1) - Tính pHash phát hiện trùng lặp
*   `seaborn` & `matplotlib` - Trực quan hóa dữ liệu
*   `scipy` - Kiểm định ANOVA, KS-Test

*Xử lý dữ liệu dạng bảng:*

*Xử lý dữ liệu văn bản:*

---

## Dữ liệu 

### 1. Dữ liệu gốc (Raw Data)
#### **Link tải Dataset (Kaggle)**
*   **Dataset dạng ảnh:** https://www.kaggle.com/datasets/tranbao0105/lab-2-1-newplantdiseasesdataset
*   **Dataset dạng bảng:** 
*   **Dataset dạng văn bản:** https://www.kaggle.com/datasets/tranbao0105/lab-2-3

#### **Tải về và giải nén vào thư mục**
*   **Dataset dạng ảnh:** `data/raw/lab_2.1_NewPlantDiseasesDataset/`. Đảm bảo file `labels.csv` nằm trong thư mục này.
*   **Dataset dạng bảng:**
*   **Dataset dạng văn bản:**

### 2. Dữ liệu đã xử lý (Processed Data)
*   **1:** Chạy lần lượt các notebook `01_EDA_image.ipynb` và `02_preprocessing_image.ipynb`. Dữ liệu sẽ tự động được sinh ra trong thư mục `data/processed/`.
*   **2:** Tải nhanh dữ liệu đã được nhóm xử lý sẵn.
*   **Link tải Processed Data (Google Drive):** https://drive.google.com/drive/folders/1BU1vanjqZDSGPOf7bW4bnTJSD4jfXAEX?usp=sharing
*     - Folder processed_image: Bao gồm các file csv kết quả, các ảnh biểu đồ và dữ liệu ảnh sau khi tiền xử lý
*     - Folder processed_tabular:
*     _ Folder processed_text:

---

## Quy trình xử lý dữ liệu dạng ảnh

### Giai đoạn 1: Phân tích EDA (`01_EDA_image.ipynb`)
*   **Thống kê:** Tính phân phối Pixel, KDE theo kênh màu.
*   **Mất cân bằng lớp:** Kiểm tra tỉ lệ giữa lớp nhiều nhất và ít nhất.
*   **Phát hiện trùng lặp:** Sử dụng **pHash** (Perceptual Hash) để loại bỏ các ảnh trùng lặp tuyệt đối và ảnh gần trùng (Hamming Distance $\le$ 6). Phát hiện và xử lý ảnh bị gán nhãn sai (Mislabeled).
*   **Phân tích cạnh (Nâng cao):** Sử dụng Sobel, Prewitt, Canny. Kiểm định **ANOVA** chứng minh mật độ cạnh có khả năng phân biệt lớp bệnh (p-value < 0.05).

### Giai đoạn 2: Tiền xử lý & Ablation Study (`02_preprocessing_image.ipynb`)
Thực hiện thí nghiệm so sánh tác động của từng kỹ thuật:
*   **Resize:** Đánh giá mất mát thông tin qua SSIM và PSNR. Kích thước **64x64** được chọn làm tối ưu.
*   **Không gian màu:** So sánh RGB, Gray, HSV, LAB qua PCA Explained Variance. **LAB** cho kết quả tốt nhất.
*   **Chuẩn hóa:** Kiểm định **KS-Test** trên 4 phương pháp. **Z-score Per-channel** giúp mô hình đạt Accuracy cao nhất.
*   **Ablation Study:** Đánh giá hiệu năng mô hình (Logistic Regression/KNN) qua từng bước thêm mới kỹ thuật.

**Kết quả:** Kích thước 64x64 + Màu LAB + Z-score chuẩn hóa. (Lưu ý: Augmentation không được áp dụng cho pipeline cuối do làm giảm hiệu năng trên mô hình vector hóa đơn giản).

## Chi tiết quy trình xử lý dữ liệu dạng bảng

## Chi tiết quy trình xử lý dữ liệu văn bản

---

## Hướng dẫn chạy Notebook
1.  Mở thư mục dự án bằng VS Code hoặc Jupyter Lab.
2.  Đảm bảo dữ liệu raw đã đặt đúng vị trí.
3.  Chạy **Notebook EDA** để thực hiện làm sạch dữ liệu.
4.  Chạy **Notebook Preprocessing** để thực hiện tiền xử lý và sinh ra tập dữ liệu cuối cùng.
