<!DOCTYPE html>
<html>

<head>

<title>enregistrement</title>
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8"/>


<!--Source se rediriger : http://www.redirection-web.net/redirection-html.php -->
<!--<meta http-equiv="refresh" content="durée;URL=adresse-de-destination">  -->


<?php

if($_GET['n']!="" && $_GET['p']!="" && $_GET['mail']!="" && $_GET['mdp1']!="" && $_GET['mdp2']!="" && $_GET['mdp1']== $_GET['mdp2'] ) {
	enregistrer($_GET['n'], $_GET['p'], $_GET['act'], $_GET['mail'],$_GET['mdp1']);
	echo "<meta http-equiv='refresh' content='5;URL=accueil.php'>";
	echo "<h1 class='messagePage'> Ca marche ! Bienvenue ".$_GET['p']."</h1>";
}
else{
	// permet de ne pas montrer les mot de passe saisi dans la barre url et enregistre les précedent selectionné
	echo "<meta http-equiv='refresh'; content='0;URL=nouveau.php?n=".$_GET['n']."&p=".$_GET['p']."&adr=".$_GET['adr']."&num=".$_GET['num']."&mail=".$_GET['mail']."'>";
	echo "un champs n'est pas remplis ou les mots de passe n'est pas confirmé";

}

?>


</head>


<body>

<?php

function enregistrer($nom, $prenom, $activite, $mail, $mdp){
	
	include('bd.php'); //importation de ma feuille PHP bd.php
	$bdd = getBD(); //entrée dans la BD
	$act = (int)$act;//je converti ma variable $act en integer car c'est un string. 
	$sql='INSERT INTO `user` (`id_user`, `nom`, `prenom`, `droit`, `mdp`, `mail`) VALUES (NULL, "'.$nom.'", "'.$prenom.'", '.$act.', "'.$mdp.'", "'.$mail.'")';
	$bdd->exec($sql);//on execute la requette.

}


?>


</body>
</html>