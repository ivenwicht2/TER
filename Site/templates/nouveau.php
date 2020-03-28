<!DOCTYPE html>
<html>

<head>
	<title>Nouveau</title>
	<meta http-equiv="Content-Type" content="text/html;charset=UTF-8"/>
    <title>Pixoglyphe - Accueil</title>
    <link rel="stylesheet" href="monstyle.css" />
	<?php 
	// importations de pages PHP de connexion a la BD
	include('bd.php');
	?>

</head>
<body>


    <!-- MENU -->
    <nav>
        <ul>
            <li><a href="Accueil.php">Accueil</a></li><!--
            --><li><a href="Labellisation.php">Labellisation</a></li><!--
            --><li><a href="Historique.php">Historique</a></li><!--
            --><li><a href="Bibliotheque.php">Bibliothèque</a></li>
			<?php
				session_start();
				if(isset($_SESSION['client'])){//si la session client est ouverte
					//on peut se deconnecté.
					echo '<li><a href="deconnexion.php">Déconnexion</a></li>';
				}else{
					echo '<li><a href="nouveau.php">Inscription</a></li>';//sinon on peut creer un compte
				}


			?>
        </ul>
    </nav>

 <!-- IMAGE BORD GAUCHE -->
    <img src="barre.png" width="5%" >

	
	
<div id="inscriEncadrer">
<div id="coTitle" > Inscrivez vous </div>
<form class="inscrip" method="get" action="enregistrement.php" autocomplete="off">
<table class="inscriTable">
	<tr>
		<td> Prénom </td>
		<td> <input type="text" name="p" value = "<?php if(isset($_GET['p'])) {echo $_GET['p'];} ?>"> </td>
	</tr>
	<tr>
		<td> Nom </td>
		<td> <input type="text" name="n" value = "<?php if(isset($_GET['n'])) {echo $_GET['n'];} ?>"> </td>
	</tr>
	<tr>
		<td> Adresse mail </td>
		<td> <input type="text" name="mail" value = "<?php if(isset($_GET['mail'])) {echo $_GET['mail'];} ?>"> </td>
	</tr>
	<tr>
		<td> Activité </td>
		<td> 
			<select type="select" name="act" value="">
			<OPTION value="0">Non spécialiste</OPTION>
			<OPTION value="1">Spécialiste</OPTION>
			</select>
		</td>
	</tr>
	<tr>
		<td> Mot de passe </td>
		<td> <INPUT type="password" name="mdp1" value=""/></td>
	</tr>
	<tr>
		<td> Confirmer votre mot de passe </td>
		<td> <INPUT type="password" name="mdp2" value=""/></td>
	</tr>
</table>

<input class="cobutton" type="submit" value="Insciption">

</form>


</div>



</body>
</html>