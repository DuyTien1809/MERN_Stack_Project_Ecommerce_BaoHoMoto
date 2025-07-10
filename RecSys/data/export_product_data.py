import csv
from pymongo import MongoClient

def extract_brand(product_name):
    """Hàm lấy từ cuối cùng của pName làm brand"""
    return product_name.split()[-1] if product_name else ""

def export_products_to_csv(csv_file_path, encoded_map_dir):
    # Kết nối tới MongoDB
    client = MongoClient("mongodb://localhost:27017/ecommerce")
    db = client["ecommerce"]  # Thay "your_database_name" bằng tên database của bạn
    collection = db["products"]  # Tên collection, khớp với schema productSchema

    # Lấy dữ liệu từ MongoDB
    products = collection.find({}, {"_id": 1, "pCategory": 1, "pName": 1})

    # Mã hóa các trường
    product_id_map = {}
    category_map = {}
    brand_map = {}

    product_counter = 0
    category_counter = 0
    brand_counter = 0

    # Mở file CSV để ghi
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Ghi tiêu đề cột
        writer.writerow(["product_id", "category", "brand"])

        # Duyệt qua các sản phẩm để ghi vào CSV và tạo bản đồ mã hóa
        for product in products:
            # Mã hóa product_id
            product_id = product["_id"]
            if product_id not in product_id_map:
                product_id_map[product_id] = product_counter
                product_counter += 1

            # Mã hóa category
            category = product.get("pCategory", "")
            if category not in category_map:
                category_map[category] = category_counter
                category_counter += 1

            # Mã hóa brand
            brand = extract_brand(product.get("pName", ""))
            if brand not in brand_map:
                brand_map[brand] = brand_counter
                brand_counter += 1

            # Ghi vào file CSV
            writer.writerow([product_id_map[product_id], category_map.get(category, -1), brand_map.get(brand, -1)])

    # Lưu các bản đồ mã hóa vào các file CSV
    with open(f"{encoded_map_dir}/product_id_map.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["product_id", "encoded_id"])
        for product_id, encoded_id in product_id_map.items():
            writer.writerow([product_id, encoded_id])

    with open(f"{encoded_map_dir}/category_map.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["category", "encoded_id"])
        for category, encoded_id in category_map.items():
            writer.writerow([category, encoded_id])

    with open(f"{encoded_map_dir}/brand_map.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["brand", "encoded_id"])
        for brand, encoded_id in brand_map.items():
            writer.writerow([brand, encoded_id])

    print(f"Dữ liệu đã được xuất thành công tới {csv_file_path}")
    print(f"Các bản đồ mã hóa đã được lưu vào {encoded_map_dir}")

def run_export_product_data(csv_path, encoded_map_dir):
    export_products_to_csv(csv_path, encoded_map_dir)

if __name__ == "__main__":
    csv_path = "D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/product_data.csv"  # Đường dẫn tới file CSV
    encoded_map_dir = "D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/encoded_map"  # Đường dẫn tới thư mục lưu các bản đồ mã hóa
    export_products_to_csv(csv_path, encoded_map_dir)