import React, { Fragment } from "react";
import Layout from "../layout";
import BlogPost from "./BlogPost";


const Post = () => {
  return (
    <Fragment>
    <h1>test</h1>
    <Layout children={<BlogPost/>} />
  </Fragment>
  );
};

export default Post;
