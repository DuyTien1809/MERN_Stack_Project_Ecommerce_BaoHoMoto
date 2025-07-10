// 

var braintree = require("braintree");
require("dotenv").config();

var gateway = new braintree.BraintreeGateway({
  environment: braintree.Environment.Sandbox,
  merchantId: process.env.BRAINTREE_MERCHANT_ID,
  publicKey: process.env.BRAINTREE_PUBLIC_KEY,
  privateKey: process.env.BRAINTREE_PRIVATE_KEY,
});

class brainTree {
  ganerateToken(req, res) {
    gateway.clientToken.generate({}, (err, response) => {
      if (err) {
        return res.json(err);
      }
      return res.json(response);
    });
  }

  paymentProcess(req, res) {
    let { amountTotal, paymentMethod } = req.body;
    
    // Giả sử tỷ giá hiện tại là 24,000 VND = 1 USD (hãy thay đổi nếu cần)
    const exchangeRate = 24000;

    // Chuyển đổi amountTotal từ VND sang USD
    let amountInUSD = (amountTotal / exchangeRate).toFixed(2);

    gateway.transaction.sale(
      {
        amount: amountInUSD,  // Sử dụng giá trị USD đã chuyển đổi
        paymentMethodNonce: paymentMethod,
        options: {
          submitForSettlement: true,
        },
      },
      (err, result) => {
        if (err) {
          console.error(err);
          return res.json(err);
        }

        if (result.success) {
          console.log("Transaction ID: " + result.transaction.id);
          return res.json(result);
        } else {
          console.error(result.message);
          return res.json(result);
        }
      }
    );
  }
}

const brainTreeController = new brainTree();
module.exports = brainTreeController;
