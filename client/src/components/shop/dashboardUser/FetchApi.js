import axios from "axios";
const apiURL = process.env.REACT_APP_API_URL;

export const getUserById = async (uId) => {
  try {
    let res = await axios.post(`${apiURL}/api/user/signle-user`, { uId });
    return res.data;
  } catch (error) {
    console.log(error);
  }
};

export const updatePersonalInformationFetch = async (userData) => {
  try {
    let res = await axios.post(`${apiURL}/api/user/edit-user`, userData);
    return res.data;
  } catch (error) {
    console.log(error);
  }
};

export const getOrderByUser = async (uId) => {
  try {
    let res = await axios.post(`${apiURL}/api/order/order-by-user`, { uId });
    return res.data;
  } catch (error) {
    console.log(error);
  }
};

export const updatePassword = async (formData) => {
  try {
    let res = await axios.post(`${apiURL}/api/user/change-password`, formData);
    return res.data;
  } catch (error) {
    console.log(error);
  }
};

export const forgetPassword = async (email) => {
  try {
    const res = await axios.post(`${apiURL}/api/user/forget-password`, { email });
    return res.data; // Trả về phản hồi thành công từ API
  } catch (error) {
    if (error.response && error.response.data && error.response.data.message) {     
      return { error: error.response.data.message };
    } else {   
      return { error: "An unexpected error occurred. Please try again." };
    }
  }
};