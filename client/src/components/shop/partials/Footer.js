
import React from "react";
import moment from "moment";
import {
  FaFacebook,
  FaYoutube,
  FaTiktok,
  FaCcVisa,
  FaCcMastercard,
  FaCcJcb,
} from "react-icons/fa";
import { SiZalo, SiApplepay, SiGooglepay, SiVuedotjs } from "react-icons/si";
import { AboutUsData } from "../aboutus/Information";

const Footer = () => {
  return (
    <footer style={styles.footer}>
      <div style={styles.container}>
        <div style={styles.section}>
          <h4 style={styles.heading}>KẾT NỐI VỚI H&T SHOP</h4>
          <div style={styles.socialIcons}>
            <FaFacebook size={24} color="#ffffff" />
            <SiZalo size={24} color="#ffffff" />
            <FaYoutube size={24} color="#ffffff" />
            <FaTiktok size={24} color="#ffffff" />
          </div>
          <div style={styles.hotline}>
            <p>Tư vấn mua hàng (Miễn phí)</p>
            <p>1800.6601 (Nhánh 1)</p>
            <p>Hỗ trợ kỹ thuật</p>
            <p>1800.6601 (Nhánh 2)</p>
            <p>Góp ý, khiếu nại</p>
            <p>1800.6616 (8h00 - 22h00)</p>
          </div>
        </div>

        <div style={styles.section}>
          <h4 style={styles.heading}>VỀ CHÚNG TÔI</h4>
          <ul style={styles.links}>
            
          {AboutUsData.slice(0, 2).map((item) => (
            <li key={item.id}>
              <a href={`/about-us/${item.id}`} className="link-style">{item.title}</a>
            </li>
          ))}
          </ul>
        </div>

        <div style={styles.section}>  
          <h4 style={styles.heading}>CHÍNH SÁCH</h4>
          <ul style={styles.links}>

          {AboutUsData.slice(2).map((item) => (
            <li key={item.id}>
              <a href={`/about-us/${item.id}`} className="link-style">{item.title}</a>
            </li>
          ))}
    
          </ul>
        </div>
        <div style={styles.section}>
          <h4 style={styles.heading}>HỖ TRỢ THANH TOÁN</h4>
          <div style={styles.paymentMethods}>
            <FaCcVisa size={24} color="#ffffff" />
            <FaCcMastercard size={24} color="#ffffff" />
            <FaCcJcb size={24} color="#ffffff" />
            <SiApplepay size={24} color="#ffffff" />
            <SiGooglepay size={24} color="#ffffff" />
          </div>
          <h4 style={styles.heading}>CHỨNG NHẬN</h4>
          <div style={styles.certifications}>
            <SiVuedotjs size={24} color="#ffffff" />
          </div>
        </div>
      </div>

      <div style={styles.copyright}>
        Develop & Design H&T © Copyright {moment().format("YYYY")}
      </div>
    </footer>
  );
};

const styles = {
  footer: {
    background: "#303031",
    color: "#87898A",
    padding: "20px 0",
  },
  container: {
    display: "flex",
    justifyContent: "space-around",
    flexWrap: "wrap",
    padding: "0 20px",
  },
  section: {
    flex: "1 1 200px",
    margin: "10px", 
    paddingLeft: "60px"

  },
  heading: {
    color: "#FFFFFF",
    marginBottom: "10px",
    
  },
  socialIcons: {
    display: "flex",
    gap: "10px",
  },
  hotline: {
    color: "#FFFFFF",
    fontSize: "14px",
    lineHeight: "1.5",
  },
  links: {
    listStyle: "none",
    padding: 0,
    lineHeight: "1.8",
  },
  paymentMethods: {
    display: "flex",
    gap: "10px",
    marginBottom: "10px",
  },
  certifications: {
    display: "flex",
    gap: "10px",
  },
  copyright: {
    textAlign: "center",
    paddingTop: "10px",
    color: "#87898A",
  },
};

export default Footer;
