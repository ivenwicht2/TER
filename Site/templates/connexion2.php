<?php 
include('bd.php');

echo '<div id="coEncadrer">';
echo '<div id="coTitle" > Vous devez vous connecter </div>';
echo '<form class="inscrip" method="get" action="connecter.php" autocomplete="off">';

echo '<table class="coTable">';

	echo'<tr>';
		echo'<td> Mail utilisateur </td>';
		
		echo '<td> <input type="text" name="mailCo" value ="';
		if(isset($_GET['mailCo'])) {
			echo $_GET['mailCo'];
		}
		echo'"> </td>';
		
	echo'</tr>';
	
	echo'<tr>';
		echo'<td> Mot de passe </td>';
		echo'<td> <INPUT type="password" name="passwordCo" value=""/></td>';
	echo'</tr>';
	//echo'<tr>';
		//echo'<td><input type="submit" value="Envoyer"></td>';
	//echo'</tr>';
echo'</table>';

echo '<input class="cobutton" type="submit" value="Connexion">';

echo'</form>';

echo'<a class="lienNouveau" href="nouveau.php">Je ne possede pas encore de compte</a>';

echo'</div>';

?>
