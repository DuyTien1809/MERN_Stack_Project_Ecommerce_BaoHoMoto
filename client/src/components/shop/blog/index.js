import React, { Fragment } from "react";
import Layout from "../layout";
import BlogPage from "./BlogPage";


const Blog = () => {
  return (
    <Fragment>
    <Layout children={<BlogPage/>} />
  </Fragment>
  );
};

export default Blog;
