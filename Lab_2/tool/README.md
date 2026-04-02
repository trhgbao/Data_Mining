# Chạy SPMF

Máy tính đã được cài đặt **Java**.

---

## CÁCH 1: Chạy bằng Command Line

### Cú pháp tổng quát:
```bash
java -jar spmf.jar run [Tên_Thuật_Toán] [Đường_dẫn_Input] [Đường_dẫn_Output] [Các_tham_số]
```

### Ví dụ:
Mở Terminal, di chuyển đến thư mục chứa file `spmf.jar` và chạy lệnh sau:

```bash
java -jar spmf.jar run Eclat /home/bao/Data_Mining/chess.dat /home/bao/Data_Mining/chess_result.txt 0.5
```

*   `java -jar spmf.jar run`: Lệnh gọi phần mềm SPMF thực thi thuật toán.
*   `Eclat`: Tên thuật toán bạn muốn sử dụng (Lưu ý: Phân biệt chữ hoa chữ thường).
*   `/home/bao/Data_Mining/chess.dat`: Đường dẫn tuyệt đối đến file dữ liệu đầu vào.
*   `/home/bao/Data_Mining/chess_result.txt`: Đường dẫn tuyệt đối để lưu file kết quả.
*   `0.5`: Tham số của thuật toán (Ở đây Minsup = 0.5 hay 50%).

---

## CÁCH 2: Chạy bằng Giao diện 

Click đúp chuột vào file `spmf.jar` hoặc mở Terminal và gõ:
```bash
java -jar spmf.jar
```

Tui chạy gặp lỗi không điền hay chọn được đường dẫn input, output
