import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown'; // Thư viện hiển thị Markdown
import { useParams } from 'react-router-dom';
import introcompany from './info/post1.md'
import img2 from '../blog/img/sport.jpg'
import img1 from '../blog/img/Shop.png'
import faqcontent from './info/post2.md'
import baohanh from './info/post3.md'
import doitra from './info/post4.md'
import baomat from './info/post5.md'
import vanchuyen from './info/post6.md'

export const  AboutUsData = [
  {
    id: "intro-company", 
    image: img1,
    title: "Giới thiệu về công ty",
    content: introcompany,
  },
  {
    id: "faq",  
    image: img2,
    title: "Câu hỏi thường gặp",
    content: faqcontent,
  },  
  {
    id: "chinh-sach-bao-hanh",  
    title: "Chính sách bảo hành",
    content: baohanh,
  },
  {
    id: "chinh-sach-doi-tra",  
    title: "Chính sách đổi trả",
    content: doitra,
  },
  {
    id: "chinh-sach-bao-mat",  
    title: "Chính sách bảo mật",
    content: baomat ,
  },
  {
    id: "chinh-sach-van-chuyen",  
    title: "Chính sách vận chuyển",
    content: vanchuyen,
  },

] 
const Information = () => {
  const { id } = useParams(); // Lấy ID từ URL
  const post = AboutUsData.find((item) => item.id === id); // Tìm bài viết theo ID
  const [content, setContent] = useState(''); // State lưu nội dung Markdown

  useEffect(() => { 
    if (post && post.content) {
      fetch(post.content)
        .then((response) => response.text()) // Đọc nội dung file Markdown
        .then((text) => setContent(text)) // Lưu vào state
        .catch((error) => console.error('Error loading Markdown:', error));
    }
  }, [post]);

  if (!post) {
    return <div>Post not found!</div>;
  }

  return (
    <div style={styles.background}>
      <div style={styles.contentWrapper}>
        <h1 style={styles.title}>{post.title}</h1>
        <div style={styles.content}>
          {/* Render phần nội dung trước ảnh */}
          <ReactMarkdown>
            {content.split("<!-- image -->")[0]}
          </ReactMarkdown>
  
          {/* Kiểm tra nếu có hình thì mới hiển thị */}
          {post.image && (
            <img src={post.image} alt={post.title} style={styles.image} />
          )}
  
          {/* Render phần nội dung sau ảnh nếu có */}
          {content.includes("<!-- image -->") && (
            <ReactMarkdown>
              {content.split("<!-- image -->")[1]}
            </ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  );
  
  
};
const styles = {
  background: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    minHeight: "0vh",
    backgroundColor: "#f5f5dc",
    paddingTop: "8%",
    paddingBottom: "40px", // Khoảng cách với footer
    flexDirection: "column",
  },
  contentWrapper: {
    width: "90%",
    maxWidth: "900px",
    backgroundColor: "#ffffff",
    padding: "20px",
    borderRadius: "8px",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
  },
  title: {
    fontSize: "32px",
    fontWeight: "bold",
    marginBottom: "20px",
    textAlign: "center",
    color: "#333333",
  },
  image: {
    width: "100%", 
    height: "auto",
    objectFit: "cover",
    borderRadius: "10px",
    display: "block",       
    marginLeft: "auto",     
    marginRight: "auto", 
    marginBottom: "40px"
   
  },
  content: {
    textAlign: "justify",
    fontSize: "18px",
    lineHeight: "1.6",
    color: "#555555",
    wordBreak: "break-word",
    textIndent: "30px", 
    
  },
};

export default Information;
