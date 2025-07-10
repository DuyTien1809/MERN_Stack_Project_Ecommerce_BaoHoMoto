import csv

def get_encoded_id(user_id, csv_path='D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/encoded_map/user_id_map.csv'):

    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['user_id'] == user_id:
                return int(row['encoded_id'])
    raise ValueError(f"User ID {user_id} not found in mapping.")
    
