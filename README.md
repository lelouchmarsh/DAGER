# DAGER

- DAGER dùng để nhận diện độ tuổi và giới tính
- Mức chính xác giới tính là 88%
- Mức chính xác độ tuổi là 60% (Phân theo lứa tuổi)

## Dataset
- Tải Imdb và Wiki dataset tại đây: 
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- Tải UTKface dataset tại đây:  
https://susanqq.github.io/UTKFace/

## Weight:

https://drive.google.com/drive/u/1/folders/16AkVPSFDjME3grd8lQD3gLHPO6VquJXn

## Xử lý data idmb và wiki
-	Tải về, giải nén ra 2 folder là idmb_crop và wiki_crop. Ở mỗi folder có 100 folder con và 1 file mat
-	Vào Project_Gender -> Process_data -> process -> Chạy file mat.py
-	Sau khi chạy xong mat.py, ta sẽ nhận được 1 file csv có chứa ảnh, các nhãn giới tính, tuổi và địa chỉ (file meta.csv) 
-	Tạo một folder 	Gender Data
-	Để label gender, vào Project_Gender -> Process_data -> tacanh.py.
Sau đó ta sẽ nhận được 2 folder  là male và female trong folder Gender Data
-	Chạy divide.py để tạo ra 3 folder gồm train, val, test
-	Tạo một folder dataset.
-	Để label age, vào Project_Gender -> Process_data -> age.py
Sau đó ta sẽ nhận được 1 folder lớn chứa 100 folder con từ độ tuổi 1-100
-	Gộp các folder tuổi thành từng giai đoạn: 2-8, 9-14, 15-25, 26-36, 37-50, 50-65, 65-85, 85-100
## Xử lý data UTKface ( Chỉ dùng cho việc nhận diện giới tính)
- vào Process_data -> utkface.py
- Nhận được 2 folder female và male. Gộp chung với 2 folder male và female của idmb và wiki dataset
## Training
- 	Vào Project_Gender -> python getweight.py, sẽ tạo ra một file gender_weight.h5
-	python training.py
## Evaluate model 
-	Đánh giá giới tính: evalgender.py
-   Đánh giá tuổi : evalage.py
## Nhận diện từ ảnh 
Chạy ubuntu command: 
- python aligned.py [image_path]

## Nhận diện từ camera:
- python camera.py
- Nhấn Q để quit
## License