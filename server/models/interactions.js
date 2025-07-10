const mongoose = require('mongoose');

const InteractionSchema = new mongoose.Schema({
    user_id: mongoose.Schema.Types.ObjectId,
    product_id: mongoose.Schema.Types.ObjectId,
    action: {
        type: String,
        enum: ['view', 'add_to_cart','wish_list', 'purchase', 'rate'],
        required: true
    },
    rating: {
        type: Number,
        min: 1,
        max: 5,
        required: function() { return this.action === 'rate'; } // Chỉ cần rating khi action là 'rate'
    }
}, { timestamps: true });

module.exports = mongoose.model('Interaction', InteractionSchema);
