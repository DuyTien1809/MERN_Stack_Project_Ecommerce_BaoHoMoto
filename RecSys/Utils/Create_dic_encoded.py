import pandas as pd

def get_product_id_mapping():
    # Đọc file product_id_map.csv
    encoded_map = pd.read_csv('D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/encoded_map/product_id_map.csv')
    
    # Tạo dictionary ánh xạ từ encoded_id -> product_id
    encoded_to_product = dict(zip(encoded_map['encoded_id'], encoded_map['product_id']))
    
    return encoded_to_product
