const express = require('express');
const router = express.Router();
const Interaction = require('../models/interactions');  // Import model Interaction

// API để lưu hành động
router.post('/add', async (req, res) => {
    const { user_id, product_id, action, rating } = req.body;

    try {
        // Tạo một bản ghi mới
        const interaction = new Interaction({ user_id, product_id, action, rating });
        await interaction.save();

        res.status(201).json({ message: 'Hành động đã được lưu thành công', interaction });
    } catch (error) {
        res.status(500).json({ message: 'Lỗi khi lưu hành động', error });
    }
});

module.exports = router;
