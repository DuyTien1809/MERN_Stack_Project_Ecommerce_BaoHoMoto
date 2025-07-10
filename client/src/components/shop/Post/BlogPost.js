import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown'; // Thư viện hiển thị Markdown
import { useParams } from 'react-router-dom';
import { blogData } from '../blog/BlogPage';

const BlogPost = () => {
  const { id } = useParams(); // Lấy ID từ URL
  const post = blogData.find((item) => item.id === parseInt(id)); // Tìm bài viết theo ID
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
        <img src={post.image} alt={post.title} style={styles.image} />
        <div style={styles.content}>
          <ReactMarkdown>{content}</ReactMarkdown>
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
    backgroundColor: "#f4f4f4",
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

export default BlogPost;
