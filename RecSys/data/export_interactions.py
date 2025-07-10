# import csv
# from pymongo import MongoClient


# def load_product_id_map(encoded_map_dir):
#     """Hàm tải file mã hóa product_id từ thư mục đã chỉ định"""
#     product_id_map = {}
#     try:
#         with open(f"{encoded_map_dir}/product_id_map.csv", mode='r', encoding='utf-8') as file:
#             reader = csv.reader(file)
#             next(reader)  # Bỏ qua tiêu đề cột
#             for row in reader:
#                 product_id_map[row[0]] = int(row[1])
#     except FileNotFoundError:
#         print(f"File {encoded_map_dir}/product_id_map.csv không tồn tại.")
#     return product_id_map

# def export_interactions_to_csv(csv_file_path, encoded_map_dir):
#     # Kết nối tới MongoDB
#     client = MongoClient("mongodb://localhost:27017/ecommerce")
#     db = client["ecommerce"]  
#     collection = db["interactions"]  

#     # Lấy dữ liệu từ MongoDB
#     interactions = collection.find({}, {"user_id": 1, "product_id": 1, "action": 1, "_id": 0})

#     # Mã hóa các trường user_id, action
#     user_id_map = {}
#     action_map = {}

#     user_counter = 0
#     action_counter = 0

#     # Mã hóa product_id từ file product_id_map.csv
#     product_id_map = load_product_id_map(encoded_map_dir)

#     # Mở file CSV để ghi
#     with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         # Ghi tiêu đề cột
#         writer.writerow(["user_id", "product_id", "action"])

#         # Duyệt qua các tương tác để mã hóa và ghi vào CSV
#         for interaction in interactions:
#             # Mã hóa user_id
#             user_id = interaction["user_id"]
#             if user_id:
#                 if user_id not in user_id_map:
#                     user_id_map[user_id] = user_counter
#                     user_counter += 1
            
            
#             # Mã hóa action
#             action = interaction["action"]
#             if action not in action_map:
#                 action_map[action] = action_counter
#                 action_counter += 1

#             # Mã hóa product_id (dựa trên product_id_map)
#             product_id = interaction["product_id"]
#             encoded_product_id = product_id_map.get(str(product_id), -1)  # Nếu không tìm thấy mã hóa thì gán -1

#             # Ghi vào file CSV
#             writer.writerow([user_id_map[user_id], encoded_product_id, action_map.get(action, -1)])

#     # Lưu các bản đồ mã hóa vào các file CSV
#     with open(f"{encoded_map_dir}/user_id_map.csv", mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(["user_id", "encoded_id"])
#         for user_id, encoded_id in user_id_map.items():
#             writer.writerow([user_id, encoded_id])

#     with open(f"{encoded_map_dir}/action_map.csv", mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(["action", "encoded_id"])
#         for action, encoded_id in action_map.items():
#             writer.writerow([action, encoded_id])

#     print(f"Dữ liệu đã được xuất thành công tới {csv_file_path}")
#     print(f"Các bản đồ mã hóa đã được lưu vào {encoded_map_dir}")

# def run_export_interact(csv_path,encoded_map_dir):
 
#     export_interactions_to_csv(csv_path, encoded_map_dir)

# if __name__ == "__main__":
#     csv_path = "D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/interactions_data.csv"  
#     encoded_map_dir = "D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/encoded_map"  
#     export_interactions_to_csv(csv_path, encoded_map_dir)

import csv
from pymongo import MongoClient

def load_product_id_map(encoded_map_dir):
    """Hàm tải file mã hóa product_id từ thư mục đã chỉ định"""
    product_id_map = {}
    try:
        with open(f"{encoded_map_dir}/product_id_map.csv", mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua tiêu đề cột
            for row in reader:
                product_id_map[row[0]] = int(row[1])
    except FileNotFoundError:
        print(f"File {encoded_map_dir}/product_id_map.csv không tồn tại.")
    return product_id_map

def export_interactions_to_csv(csv_file_path, encoded_map_dir):
    # Kết nối tới MongoDB
    client = MongoClient("mongodb://localhost:27017/ecommerce")
    db = client["ecommerce"]  
    collection = db["interactions"]  

    # Lấy dữ liệu từ MongoDB
    interactions = collection.find({}, {"user_id": 1, "product_id": 1, "action": 1, "_id": 0})

    # Mã hóa các trường user_id, action
    user_id_map = {}
    action_map = {}

    user_counter = 0
    action_counter = 0

    # Mã hóa product_id từ file product_id_map.csv
    product_id_map = load_product_id_map(encoded_map_dir)

    # Mở file CSV để ghi
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Ghi tiêu đề cột
        writer.writerow(["user_id", "product_id", "action"])

        # Duyệt qua các tương tác để mã hóa và ghi vào CSV
        for interaction in interactions:
            # Kiểm tra sự tồn tại của trường user_id
            if "user_id" not in interaction or not interaction["user_id"]:
                continue  # Bỏ qua interaction này nếu không có user_id

            # Mã hóa user_id
            user_id = interaction["user_id"]
            if user_id not in user_id_map:
                user_id_map[user_id] = user_counter
                user_counter += 1
            
            
            # Mã hóa action
            action = interaction["action"]
            if action not in action_map:
                action_map[action] = action_counter
                action_counter += 1

            # Mã hóa product_id (dựa trên product_id_map)
            product_id = interaction["product_id"]
            encoded_product_id = product_id_map.get(str(product_id), -1)  # Nếu không tìm thấy mã hóa thì gán -1

            # Ghi vào file CSV
            writer.writerow([user_id_map[user_id], encoded_product_id, action_map.get(action, -1)])

    # Lưu các bản đồ mã hóa vào các file CSV
    with open(f"{encoded_map_dir}/user_id_map.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "encoded_id"])
        for user_id, encoded_id in user_id_map.items():
            writer.writerow([user_id, encoded_id])

    with open(f"{encoded_map_dir}/action_map.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["action", "encoded_id"])
        for action, encoded_id in action_map.items():
            writer.writerow([action, encoded_id])

    print(f"Dữ liệu đã được xuất thành công tới {csv_file_path}")
    print(f"Các bản đồ mã hóa đã được lưu vào {encoded_map_dir}")

def run_export_interact(csv_path,encoded_map_dir):
 
    export_interactions_to_csv(csv_path, encoded_map_dir)

if __name__ == "__main__":
    csv_path = "D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/interactions_data.csv"  
    encoded_map_dir = "D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/encoded_map"  
    export_interactions_to_csv(csv_path, encoded_map_dir)