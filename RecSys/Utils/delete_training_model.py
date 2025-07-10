import os
import glob

def delete_csv_files():
    # Danh sách các thư mục cần kiểm tra
    directories = ['D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data',
     'D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/encoded_map']
    
    for directory in directories:
        # Tìm tất cả các file .csv trong thư mục
        csv_files = glob.glob(os.path.join(directory, '*.csv'))
        
        if not csv_files:
            print(f"Không có file .csv nào trong thư mục: {directory}")

        for file in csv_files:
            try:
                os.remove(file)
                print(f"Đã xóa: {file}")
            except Exception as e:
                print(f"Lỗi khi xóa file {file}: {e}")

if __name__ == "__main__":
    delete_csv_files()
