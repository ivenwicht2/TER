<?php

function getBD(){
	$bdd = new PDO('mysql:host=localhost;dbname=TERtest1;charset=utf8', 'root', 'root');
	return $bdd;
}

?>