const mongoose = require("mongoose");
try {
  // mongoose.connect("mongodb://localhost:27017/Ecommerce", {
  //   useNewUrlParser: true,
  //   useUnifiedTopology: true,
  //   useCreateIndex: true,
  // });
  mongoose.connect("mongodb://admin:secret@localhost:27017/ecommerce", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    useCreateIndex: true,
  });
  console.log("Database Connected Successfully");
} catch (err) {
  console.log("Database Not Connected");
}
