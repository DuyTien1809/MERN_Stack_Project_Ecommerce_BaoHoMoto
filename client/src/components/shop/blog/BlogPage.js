import React from "react";
import { Link } from "react-router-dom";
import img1 from "./img/phụ tụng q11.jpg";
import img2 from "./img/áo mưa.webp";
import img3 from "./img/vệ sinh giáp bảo hộ.jpg";
import img4 from "./img/dung-dich-duong-sen-ix50-quan-12.jpg";
import img5 from "./img/áo giáp alpi.jpg";
import img6 from "./img/ao-alpinestars-atem.jpg";
import img7 from "./img/do-di-phuot-hcm.jpg";
import img8 from "./img/vat-dung-can-thiet-di-phuot.jpg";
import post1Content from "./content/post1.md";
import post2Content from "./content/post2.md";
import post3Content from "./content/post3.md";
import post4Content from "./content/post4.md";
import post5Content from "./content/post5.md";
import post6Content from "./content/post6.md";
import post7Content from "./content/post7.md";
import post8Content from "./content/post8.md";
export const blogData = [
  {
    id: 1,  
    image: img1,
    title: "ĐẠI LÝ GIVI NGAY GẦN SÂN VẬN ĐỘNG PHÚ THỌ QUẬN 11",
    content: post1Content,
  
  },
  {
    id: 2,
    image: img2,
    title: "CÁCH GIẶT VÀ BẢO QUẢN ÁO MƯA TRONG NHỮNG NGÀY MƯA",
    content: post2Content,
  },
  {
    id: 3,
    image: img3,
    title: "CÁCH VỆ SINH GIÁP BẢO HỘ MOTO TẠI NHÀ MÀ AI CŨNG CÓ THỂ LÀM ĐƯỢC",
    content: post3Content,
  },
  {
    id: 4,
    image: img4,
    title: "XỊT SÊN IX50 TẠI QUẬN 12 NGAY CẦU VƯỢT QUANG TRUNG",
    content: post4Content,
  },
  {
    id: 5,
    image: img5,
    title: "ÁO GIÁP ALPINESTARS CHÍNH HÃNG VÌ SAO ĐƯỢC ƯU ÁI KHI ĐI PHƯỢT ",
    content: post5Content,
  },
  {
    id: 6,
    image: img6,
    title:"TÌM HIỂU VỀ CÁC DÒNG ÁO ALPINESTARS CHÍNH HẢNG ĐANG CÓ MẶT TẠI VIỆT NAM",
    content: post6Content,
    
  },
  {
    id: 7,
    image: img7,
    title: "ĐỒ ĐI PHƯỢT VÀ ĐỊA CHỈ CUNG CẤP UY TÍN NÊN BIẾT !!",
    content: post7Content,
  },
  {
    id: 8,
    image: img8,
    title: "NHỮNG VẬT DỤNG KHÔNG THỂ THIẾU VÀ KINH NGHIỆM CHUẨN BỊ ĐỒ TRƯỚC KHI ĐI PHƯỢT",
    content: post8Content,
  },
];
const BlogPage = () => {
    return (
      <div style={styles.background}>
        <div style={styles.container}>
          {blogData.map((post) => (
            <div key={post.id} style={styles.card}>
              <Link to={`/blog/${post.id}`} style={{ textDecoration: 'none' }}>
                <div style={styles.imgContainer}>
                  <img src={post.image} alt={post.title} style={styles.image} />
                </div>
                
                <h3 style={styles.title}>{post.title}</h3>
              </Link>
            </div>
          ))}
        </div>
      </div>
    );
  };
  

  const styles = {
    background: {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      minHeight: "100vh",
      backgroundColor: "#f5f5dc",
      paddingTop: "120px",
      flexDirection: "column", 
    },
    container: {
      display: "flex",
      flexWrap: "wrap",
      gap: "15px",
      justifyContent: "center",
    },
    card: {
      width: "300px",
      backgroundColor: "#333",
      color: "#fff",
      padding: "10px",
      borderRadius: "8px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      textAlign: "center",
      marginBottom: "30px", 
      height: "350px", 
      boxSizing: "border-box",
    },
    imgContainer: {
      width: "100%",
      height: "250px", // Chỉnh chiều cao cố định cho vùng chứa ảnh
      overflow: "hidden", // Đảm bảo ảnh không bị tràn ra ngoài
      borderRadius: "8px",
      marginBottom: "10px",
    },
    image: {
      width: "100%", // Đảm bảo ảnh chiếm hết chiều rộng của thẻ div
      height: "100%", // Đảm bảo ảnh có chiều cao đầy đủ trong container
      objectFit: "cover", // Giữ tỷ lệ ảnh mà không làm biến dạng
    },
    title: {
      fontSize: "16px",
      fontWeight: "bold",
      marginTop: "10px", // Thêm khoảng cách với ảnh
    },
  };
  
export default BlogPage;
