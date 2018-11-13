Mô tả
=============
## Chuẩn bị dữ liệu:
File dữ liệu dùng để train là: PDNA-543_sequence.fasta. Trong đó với mỗi record ta có một sequence là chuỗi protein.
File nhãn dùng để train là: PDNA-543_label.fasta. Trong đó tương ứng với mỗi phần tử protein đều được đánh nhãn 0 hoặc 1.
Để chuẩn bị dữ liệu train, sử dụng phương pháp slide window: (default window size = 10)
- Đọc tất cả sequence và nhãn tương ứng từ 2 file trên. Sau đó cắt ra sequence và label tương ứng rồi lưu vào một dict có dạng sau:
data_dict = {'sequence': label} (label được chuyển về list của integer)
- Sau đó lưu dict này vào sample_dataset/data_dna.pkl
## Cách train mô hình:
Để train mô hình, chạy file launch.bat.
Train mô hình sử dụng k-fold (default k = 5)
- Chọn 1 giá trị cho seed (0 <= seed < k) sau đó chia tập dữ liệu train theo seed, 4 phần để train, 1 phần để validate.
- Dữ liệu được chia theo index và phân thành 2 file train_ids.csv và val_ids.csv (ví dụ: sequence 1 dùng để train thì index 1 được ghi vào file train_ids.csv)
- Sau đó dữ liệu được feed vào mô hình theo từng batch, mô hình được train với 50 epochs cho mỗi fold.
## Cấu trúc mạng:
- 1 conv layer với batchnorm, relu và arvpooling.
- 1 dense layer với softmax 
## Cách tạo embedding cho protein sequence:
Tạo theo [DeepBind] (https://media.nature.com/original/nature-assets/nbt/journal/v33/n8/extref/nbt.3300-S2.pdf)

Kết quả:
--------
- Chọn 1 giá trị threshold. Khi validate, cho sequence của tập val qua network, output là một vector có độ dài bằng với độ dài của sequence. Giá trị của mỗi phần tử trong vector đó nếu nhỏ hơn threshold thì đặt bằng 0, ngược lại đặt bằng 1. Sau đó so sánh dự đoán này với label tương ứng của sequence đó.
- Kết quả xem trong checkpoints. (Với setup tham số xem trong file opt_train.txt)
