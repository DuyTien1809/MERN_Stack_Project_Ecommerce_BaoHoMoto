import React, { useEffect, useState } from "react";
import axios from "axios";
import { SiZalo, SiGmail } from "react-icons/si";

const apiURL = process.env.REACT_APP_API_URL;

const ContactUsPage = () => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false); // Thêm trạng thái xử lý gửi

  const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isSubmitting) return; // Nếu đang gửi, không thực hiện thêm
    setIsSubmitting(true);

    if (!validateEmail(email)) {
      setMessage("Email không hợp lệ. Vui lòng nhập lại.");
      setIsSubmitting(false); // Bật lại nút gửi
      return;
    }

    try {
      const response = await axios.post(`${apiURL}/send-email`, {
        toEmail: email,
      });
      setMessage(response.data.message);
      setEmail('');
      setIsSubmitted(true);
      setTimeout(() => {
        setMessage('');
        setIsSubmitted(false);
      }, 3000);
    } catch (error) {
      console.error('Lỗi khi gửi email:', error);
      setMessage(error.response?.data?.message || "Có lỗi xảy ra khi gửi email. Vui lòng thử lại.");
    } finally {
      setIsSubmitting(false); // Bật lại nút gửi
    }
  };

  useEffect(() => {
    // Thêm Google Map vào trang
    function addGoogleMap() {
      const iframe = document.createElement("iframe");
      iframe.src = "http://surl.li/stenda"; // Không thay đổi URL
      iframe.width = "100%";
      iframe.height = "100%";
      iframe.style.border = "0";
      iframe.allowFullscreen = true;
      iframe.loading = "lazy";
      iframe.referrerPolicy = "no-referrer-when-downgrade";
      document.getElementById("map-container").appendChild(iframe);
    }
    addGoogleMap();

    return () => {
      const mapContainer = document.getElementById("map-container");
      if (mapContainer) {
        mapContainer.innerHTML = ""; // Xóa iframe
      }
    };
  }, []);

  return (
    <div style={styles.background}>
      <div style={styles.container}>
        <div id="map-container" style={styles.map}></div>
        <div style={styles.form}>
          <h2 style={styles.heading}>Contact Us</h2>
          <form style={styles.formContent} onSubmit={handleSubmit}>
            <label style={{ display: "flex", alignItems: "center", ...styles.label }}>
              <SiGmail size={30} color="red" style={{ marginRight: "5px" }} />
              Gmail
            </label>
            <input
              placeholder="Enter your email"
              style={styles.input}
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <button style={styles.button} type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Đang gửi..." : "Gửi yêu cầu"}
            </button>

            {isSubmitted && <p style={{ color: "green", textAlign: "center" }}>{message}</p>}
            {!isSubmitted && message && <p style={{ color: "red", textAlign: "center" }}>{message}</p>}

            <p style={{ fontStyle: "italic", textAlign: "center", margin: "10px 0" }}>
              hoặc liên hệ trực tiếp qua Zalo
            </p>
            <a
              href="https://zalo.me/0818181948"
              target="_blank"
              rel="noopener noreferrer"
              style={{ display: "flex", justifyContent: "center" }}
            >
              <div style={styles.iconContainerStyle}>
                <SiZalo size={45} color="blue" />
              </div>
            </a>
          </form>
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
    height: "100vh",
    backgroundColor: "#f5f5dc",
  },
  container: {
    display: "flex",
    flexDirection: "row",
    width: "80%",
    height: "80%",
    backgroundColor: "rgba(255, 255, 255, 0.9)",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
    borderRadius: "10px",
    overflow: "hidden",
    marginTop: "100px",
  },
  map: {
    flex: 1,
    height: "100%",
  },
  form: {
    flex: 1,
    padding: "40px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "stretch",
    backgroundColor: "#f9f9f9",
    borderRadius: "10px",
  },
  heading: {
    fontSize: "24px",
    fontWeight: "600",
    color: "#333",
    marginBottom: "20px",
    textAlign: "center",
  },
  formContent: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
  },
  label: {
    fontSize: "20px",
    fontWeight: "500",
    color: "#555",
  },
  input: {
    width: "100%",
    padding: "12px",
    marginTop: "5px",
    border: "1px solid #ddd",
    borderRadius: "5px",
    fontSize: "14px",
    outline: "none",
  },
  button: {
    marginTop: "20px",
    padding: "12px",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
    fontWeight: "500",
  },
  iconContainerStyle: {
    display: 'inline-flex',
    justifyContent: 'center',
    alignItems: 'center',
    width: '60px',
    height: '60px',
    border: '2px solid blue',
    borderRadius: '10px',
    padding: '5px',
    backgroundColor: 'white',
  }
};

export default ContactUsPage;
