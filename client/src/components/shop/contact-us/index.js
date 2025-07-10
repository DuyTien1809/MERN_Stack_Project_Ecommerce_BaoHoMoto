import React, { Fragment } from "react";
import Layout from "../layout";
import ContactUsPage from "./ContactUsPage";


const ContactUs = () => {
  return (
    <Fragment>
    <Layout children={<ContactUsPage />} />
  </Fragment>
  );
};

export default ContactUs;
