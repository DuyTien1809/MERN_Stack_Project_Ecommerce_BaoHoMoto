
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Đọc dữ liệu từ các file CSV
interactions = pd.read_csv('D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/interactions_data.csv')
products = pd.read_csv('D:/HaoHaoHao/Hoc Hanh/HK7/Phát triển các hệ thống thông minh/Đồ án môn thông minh/E-Commerce_DoBaoHo/RecSys/data/product_data.csv')

# Mã hóa dữ liệu hành động (actions) với độ ưu tiên
action_priority = {0: 1, 1: 3, 2: 5, 3: 4}  # view: 1, add_to_cart: 3, purchase: 5, wish_list: 4

# Tạo ma trận tương tác của người dùng và sản phẩm

interactions['priority'] = interactions['action'].map(action_priority)
user_product_matrix = interactions.groupby(['user_id', 'product_id'])['priority'].sum().unstack(fill_value=0)


# Phần đề xuất sản phẩm theo hành động người dùng (KNN dựa trên hành động)
def knn_based_recommendations(user_id, user_product_matrix, k=5):
    print('user_product_matrix')
    print(user_product_matrix)
    if isinstance(user_product_matrix, pd.DataFrame):
        if len(user_product_matrix) <= k:
            k = len(user_product_matrix) - 1 
        
        user_data = user_product_matrix.loc[user_id]
        
        model_knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        model_knn.fit(user_product_matrix)
        distances, indices = model_knn.kneighbors([user_data])
        
        similar_users = user_product_matrix.index[indices.flatten()]
        recommended_products = []
        
        for similar_user in similar_users:
            products_interacted = user_product_matrix.loc[similar_user][user_product_matrix.loc[similar_user] > 0].index
            recommended_products.extend(products_interacted)
        
        recommended_products = list(set(recommended_products) - set(user_data[user_data > 0].index))
        
        return recommended_products
    else:
        raise ValueError("user_product_matrix phải là một DataFrame của Pandas")

# Phần đề xuất sản phẩm theo độ tương đồng giữa các sản phẩm (Content-Based Filtering)
def content_based_recommendations(user_id, k=5):
    product_features = products[['category', 'brand']]
    product_features = pd.get_dummies(product_features)
    
    similarity_matrix = cosine_similarity(product_features)
    
    interacted_products = user_product_matrix.loc[user_id][user_product_matrix.loc[user_id] > 0].index
    
    scores = np.zeros(len(products))
    for product_id in interacted_products:
        product_idx = products[products['product_id'] == product_id].index[0]
        scores += similarity_matrix[product_idx]
    
    recommended_indices = np.argsort(scores)[::-1][:k]
    recommended_products = products.iloc[recommended_indices]['product_id'].values
    
    recommended_products = [product for product in recommended_products if product not in interacted_products]
    
    return recommended_products

# Hàm lấy sản phẩm có nhiều lượt tương tác nhất
def get_most_interacted_products(interactions_df, k=4):
    product_interaction_counts = interactions_df.groupby('product_id')['priority'].sum()
    most_interacted_products = product_interaction_counts.sort_values(ascending=False).head(k).index.tolist()
    return most_interacted_products

# Hàm chính để đưa ra đề xuất thông minh kết hợp, giới hạn 4 sản phẩm
def combined_recommendation(user_id, k=4):
    knn_recommendations = knn_based_recommendations(user_id, user_product_matrix, k)
    content_recommendations = content_based_recommendations(user_id, k)
    most_interacted_recommendations = get_most_interacted_products(interactions, k)

    print(f"Đề xuất KNN cho người dùng {user_id}: {knn_recommendations}")
    print(f"Đề xuất Content-based cho người dùng {user_id}: {content_recommendations}")
    print(f"Đề xuất dựa trên sản phẩm có lượt tương tác nhiều nhất cho người dùng {user_id}: {most_interacted_recommendations}")

    recommendations = []

    # Ưu tiên 1: KNN
    recommendations.extend(knn_recommendations)
    print('first',recommendations)

    # Ưu tiên 2: Content-based, chỉ thêm nếu chưa đủ 4 sản phẩm
    for rec in content_recommendations:
        if rec not in recommendations:
            recommendations.append(rec)
    print('second',recommendations)    

    # Ưu tiên 3: Most interacted, chỉ thêm nếu chưa đủ 4 sản phẩm
    for rec in most_interacted_recommendations:
        if rec not in recommendations:
            recommendations.append(rec)
    print('third',recommendations[:k])
    
    return recommendations[:k]

if __name__ == "__main__":
    user_id = 0
    recommendations = combined_recommendation(user_id,k=4)
    print(f"Đề xuất cho người dùng {user_id}: {recommendations}")