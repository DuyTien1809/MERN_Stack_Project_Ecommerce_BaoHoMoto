// import React, { Fragment, useState, useContext } from "react";
// import { loginReq } from "./fetchApi";
// import { LayoutContext } from "../index";
// import { useSnackbar } from 'notistack';
// const Login = (props) => {
//   const { data: layoutData, dispatch: layoutDispatch } = useContext(
//     LayoutContext
//   );

//   const [data, setData] = useState({
//     email: "",
//     password: "",
//     error: false,
//     loading: true,
//   });

//   const alert = (msg) => <div className="text-xs text-red-500">{msg}</div>;

//   const { enqueueSnackbar } = useSnackbar();

//   const formSubmit = async () => {
//     setData({ ...data, loading: true });
//     try {
//       let responseData = await loginReq({
//         email: data.email,
//         password: data.password,
//       });
//       if (responseData.error) {
//         setData({
//           ...data,
//           loading: false,
//           error: responseData.error,
//           password: "",
//         });
//       } else if (responseData.token) {
//         setData({ email: "", password: "", loading: false, error: false });
//         localStorage.setItem("jwt", JSON.stringify(responseData));
//        enqueueSnackbar('Login Completed Successfully..!', { variant: 'success' })
//         window.location.href = "/";

//       }    
//     } catch (error) {
//       console.log(error);
//     }
//   };

//   return (
//     <Fragment>
//       <div className="text-center text-2xl mb-6">Login</div>
//       {layoutData.loginSignupError ? (
//         <div className="bg-red-200 py-2 px-4 rounded">
//           You need to login for checkout. Haven't accont? Create new one.
//         </div>
//       ) : (
//         ""
//       )}
//       <form className="space-y-4">
//         <div className="flex flex-col">
//           <label htmlFor="name">
//             Username or email address
//             <span className="text-sm text-gray-600 ml-1">*</span>
//           </label>
//           <input
//             onChange={(e) => {
//               setData({ ...data, email: e.target.value, error: false });
//               layoutDispatch({ type: "loginSignupError", payload: false });
//             }}
//             value={data.email}
//             type="text"
//             id="name"
//             className={`${
//               !data.error ? "" : "border-red-500"
//             } px-4 py-2 focus:outline-none border`}
//           />
//           {!data.error ? "" : alert(data.error)}
//         </div>
//         <div className="flex flex-col">
//           <label htmlFor="password">
//             Password<span className="text-sm text-gray-600 ml-1">*</span>
//           </label>
//           <input
//             onChange={(e) => {
//               setData({ ...data, password: e.target.value, error: false });
//               layoutDispatch({ type: "loginSignupError", payload: false });
//             }}
//             value={data.password}
//             type="password"
//             id="password"
//             className={`${
//               !data.error ? "" : "border-red-500"
//             } px-4 py-2 focus:outline-none border`}
//           />
//           {!data.error ? "" : alert(data.error)}
//         </div>
//         <div className="flex flex-col space-y-2 md:flex-row md:justify-between md:items-center">
//           <div>
//             <input
//               type="checkbox"
//               id="rememberMe"
//               className="px-4 py-2 focus:outline-none border mr-1"
//             />
//             <label htmlFor="rememberMe">
//               Remember me<span className="text-sm text-gray-600">*</span>
//             </label>
//           </div>
//           <a className="block text-gray-600" href="/">
//             Lost your password?
//           </a>
//         </div>
//         <div
//           onClick={(e) => formSubmit()}
//           style={{ background: "#303031" }}
//           className="font-medium px-4 py-2 text-white text-center cursor-pointer"
//         >
//           Login
//         </div>
//       </form>
//     </Fragment>
//   );
// };

// export default Login;

import React, { Fragment, useState, useContext } from "react";
import { loginReq } from "./fetchApi";
import { LayoutContext } from "../index";
import { useSnackbar } from "notistack";
import { forgetPassword } from "../dashboardUser/FetchApi";

const Login = (props) => {
  const { data: layoutData, dispatch: layoutDispatch } =
    useContext(LayoutContext);
  const [data, setData] = useState({
    email: "",
    password: "",
    error: false,
    loading: true,
  });
  const [isForgotPassword, setIsForgotPassword] = useState(false);
  const [resetEmail, setResetEmail] = useState(""); // Email dùng trong ForgetPassword
  const { enqueueSnackbar } = useSnackbar();

  const formSubmit = async () => {
    setData({ ...data, loading: true });
    try {
      let responseData = await loginReq({
        email: data.email,
        password: data.password,
      });
      if (responseData.error) {
        setData({
          ...data,
          loading: false,
          error: responseData.error,
          password: "",
        });
      } else if (responseData.token) {
        setData({ email: "", password: "", loading: false, error: false });
        localStorage.setItem("jwt", JSON.stringify(responseData));
        enqueueSnackbar("Login Completed Successfully..!", {
          variant: "success",
        });
        window.location.href = "/";
      }
    } catch (error) {
      console.log(error);
    }
  };

  const handleForgotPasswordSubmit = async () => {
    try {
      const response = await forgetPassword(resetEmail); // Gọi API forgetPassword
      if (response.error) {
        // Hiển thị lỗi nếu API trả về error
        enqueueSnackbar(response.error, { variant: "error" });
      } else if (response.message) {
        // Hiển thị thông báo thành công
        enqueueSnackbar(response.message, { variant: "success" });
        setIsForgotPassword(false); // Quay lại form login
      } else {
        // Thông báo lỗi bất ngờ
        enqueueSnackbar("Unexpected error occurred. Please try again.", {
          variant: "error",
        });
      }
    } catch (error) {
      // Hiển thị lỗi khi không kết nối được server
      enqueueSnackbar("Failed to connect to the server. Please try again.", {
        variant: "error",
      });
    }
  };
  

  return (
    <Fragment>
      <div className="text-center text-2xl mb-6">
        {isForgotPassword ? "Forgot Password" : "Login"}
      </div>
      {layoutData.loginSignupError ? (
        <div className="bg-red-200 py-2 px-4 rounded">
          You need to login for checkout. Haven't account? Create new one.
        </div>
      ) : null}

      {!isForgotPassword ? (
        // Form đăng nhập
        <form className="space-y-4">
          <div className="flex flex-col">
            <label htmlFor="email">
              Username or email address
              <span className="text-sm text-gray-600 ml-1">*</span>
            </label>
            <input
              onChange={(e) => {
                setData({ ...data, email: e.target.value, error: false });
                layoutDispatch({ type: "loginSignupError", payload: false });
              }}
              value={data.email}
              type="text"
              id="email"
              className={`${
                !data.error ? "" : "border-red-500"
              } px-4 py-2 focus:outline-none border`}
            />
            {!data.error ? (
              ""
            ) : (
              <div className="text-xs text-red-500">{data.error}</div>
            )}
          </div>
          <div className="flex flex-col">
            <label htmlFor="password">
              Password<span className="text-sm text-gray-600 ml-1">*</span>
            </label>
            <input
              onChange={(e) => {
                setData({ ...data, password: e.target.value, error: false });
                layoutDispatch({ type: "loginSignupError", payload: false });
              }}
              value={data.password}
              type="password"
              id="password"
              className={`${
                !data.error ? "" : "border-red-500"
              } px-4 py-2 focus:outline-none border`}
            />
            {!data.error ? (
              ""
            ) : (
              <div className="text-xs text-red-500">{data.error}</div>
            )}
          </div>
          <div className="flex flex-col space-y-2 md:flex-row md:justify-between md:items-center">
            <div>
              <input
                type="checkbox"
                id="rememberMe"
                className="px-4 py-2 focus:outline-none border mr-1"
              />
              <label htmlFor="rememberMe">
                Remember me<span className="text-sm text-gray-600">*</span>
              </label>
            </div>
            <button
              type="button"
              onClick={() => setIsForgotPassword(true)}
              className="text-gray-600"
            >
              Lost your password?
            </button>
          </div>
          <div
            onClick={formSubmit}
            style={{ background: "#303031" }}
            className="font-medium px-4 py-2 text-white text-center cursor-pointer"
          >
            Login
          </div>
        </form>
      ) : (
        // Form quên mật khẩu
        <form className="space-y-4">
          <div className="flex flex-col">
            <label htmlFor="resetEmail">
              Enter your email address:
              <span className="text-sm text-gray-600 ml-1">*</span>
            </label>
            <input
              onChange={(e) => setResetEmail(e.target.value)}
              value={resetEmail}
              type="email"
              id="resetEmail"
              className="px-4 py-2 focus:outline-none border"
            />
          </div>
          <div
            onClick={handleForgotPasswordSubmit}
            style={{ background: "#303031" }}
            className="font-medium px-4 py-2 text-white text-center cursor-pointer"
          >
            Send Reset Password
          </div>
          <div
            onClick={() => setIsForgotPassword(false)}
            className="text-black text-center mt-4 px-4 py-2 border border-black cursor-pointer"
          >
            Back to Login
          </div>
        </form>
      )}
    </Fragment>
  );
};

export default Login;
