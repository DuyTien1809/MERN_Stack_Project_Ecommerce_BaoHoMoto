import React, { Fragment } from "react";
import Layout from "../layout";
import Information from "./Information";



const FooterInformation = () => {
  return (
    <Fragment>
    <Layout children={<Information/>} />
  </Fragment>
  );
};

export default FooterInformation;
