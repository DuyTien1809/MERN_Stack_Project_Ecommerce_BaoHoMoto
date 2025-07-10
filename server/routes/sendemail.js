require('dotenv').config();
const express = require('express');
const nodemailer = require('nodemailer');
const emailValidator = require('email-validator'); 
const emailExistence = require('email-existence');

const router = express.Router();

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.GMAIL_USER,
    pass: process.env.GMAIL_PASS,
  },
});
router.post("/", async (req, res) => {
    const { toEmail } = req.body;
  
    // Kiểm tra nếu email không hợp lệ
    if (!emailValidator.validate(toEmail)) {
      return res.status(400).json({ message: 'Email không hợp lệ. Vui lòng kiểm tra lại.' });
    }
  
    // Kiểm tra email tồn tại
    emailExistence.check(toEmail, async (err, exists) => {
      if (err || !exists) {
        return res.status(400).json({ message: 'Email không tồn tại. Vui lòng kiểm tra lại.' });
      }
  
      // Email hợp lệ và tồn tại, tiếp tục gửi email
      const mailOptions = {
        from: process.env.GMAIL_USER,
        to: toEmail,
        subject: 'Cảm ơn bạn đã liên hệ!',
        text: 'Cảm ơn bạn đã gửi yêu cầu! Chúng tôi sẽ phản hồi trong thời gian sớm nhất.',
      };
  
      try {
        const info = await transporter.sendMail(mailOptions);
        console.log('Email đã được gửi:', info.response);
        res.status(200).json({ message: 'Email đã được gửi thành công!' });
      } catch (error) {
        console.error('Lỗi khi gửi email:', error);
        res.status(500).json({ message: 'Không thể gửi email. Vui lòng thử lại sau.' });
      }
    });
  });
module.exports = router;
