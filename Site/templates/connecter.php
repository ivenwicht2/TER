<!DOCTYPE html>
<html>

<head>

<title>Connecter</title>
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8"/>


<?php

if($_GET['mailCo']!="" && $_GET['passwordCo']!="") {
	connexion($_GET['mailCo'], $_GET['passwordCo']);
	//echo "<meta http-equiv='refresh' content='4;URL=connexion.php'>";
	//echo "<h1 class='messagePage'> Ca marche ! RE ".$_GET['mailCo']."</h1>";
}
else{ //lorsque juste on rentre le mail et pas le mdp
	// permet de ne pas montrer les mots de passe saisi dans la barre url et enregistre les précedent selectionné
	echo "<meta http-equiv='refresh'; content='1;URL=connexion.php?mailCo=".$_GET['mailCo']."'>";
	echo "<p>un champs n'est pas remplis ou les mots de passe n'est pas confirmé</p>";

}

?>

</head>


<body>


<?php


function connexion($mailCo , $passwordCo){
	
	//echo "SUPER ON EST RENTRER DANS LA FONCTION"; 
	
	include('bd.php'); //importation de ma feuille PHP bd.php
	$bdd = getBD(); //entrée dans la BD
	$q = 'select * from user where user.mail="'.$mailCo.'" and user.mdp="'.$passwordCo.'"';//requette
	$rep= $bdd->query($q);//execution
	$ligne = $rep->fetch();//result
	
	if($ligne['nom']!=""){ // si le nom du client existe dans ma BD
		session_start();//on demare une session
		// on crée un tableau SESSION des données du client connecté
		$_SESSION['client']= array($ligne['nom'],$ligne['prenom'],$ligne['droit'],$ligne['mdp'],$ligne['mail'], $ligne['id_user']);
		echo "<meta http-equiv='refresh' content='0;URL=accueil.php'>";
		//echo "</br> ca marche le client existe";
		//echo "<h1 class='messagePage'> Vous êtes désormais connecté ".$ligne['prenom']."</h1>";
		}
	else{
		echo "<meta http-equiv='refresh'; content='4;URL=historique.php'>";
		echo "</br> <p>Les identifiant saisi ne correspond a aucun clients</p>";
		echo "</br> <p>Veuillez crée un compte </p> ";
		}
	
	
	
	
	// echo " FIN DE LA FONCTION"; 


}


?>

</body>
</html>