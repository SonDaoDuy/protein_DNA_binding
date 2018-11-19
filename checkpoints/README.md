Mo ta ket qua:
--------------

Khong nen de y den 5 metric trong thoi diem nay. Hien tai, input cua network la sequence, output cua network la 1 vector 2 chieu [a,b]. Neu a > b thi mau co nhan 0, a <= b thi mau co nhan 1. Ta co the dat them mot gia tri threshold de dieu chinh ket qua nhu mong muon (VD: a <=b and b > threshold thi moi gan nhan 1). 
(chua thu)

exp3 va exp7: 
 - dung network nhu nhau (1 conv + 1 batchnorm + 1 Relu + 1 avgpool + 1 dropout + 1 Linear)
 - khac nhau o so phan tu cua emb (input cua Linear (exp3 co 10 phan tu, exp7 co 16 phan tu))(chua test cac truong hop khac)
[exp3](https://github.com/SonDaoDuy/protein_DNA_binding/blob/master/checkpoints/exp3/results.JPG)
[exp7](https://github.com/SonDaoDuy/protein_DNA_binding/blob/master/checkpoints/exp7/results.JPG)

exp8:
 - chi dung one hot de bieu dien chuoi protein (ko dung giong trong bai bao)
 - ham loss khong hoi tu bang exp3 va exp 7
 [exp8](https://github.com/SonDaoDuy/protein_DNA_binding/blob/master/checkpoints/exp8/results.JPG)
 
exp9:
 - dung network khac (1 linear + 1 sigmoid + 1 dropout + 1 linear)
 - ham loss hoi tu khong nhanh va khong bang cac truong hop dung 1 conv
 [exp9](./exp9/results.JPG)
