<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <title>Pixoglyphe - Accueil</title>
    <link rel="stylesheet" href="monstyle.css" />
	
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

    <!--TITRE-->
    <!--<h1>Mon historique</h1>--> 
	
	<?php

	session_start();
	if(isset($_SESSION['client'])){//si la session client est ouverte
		//page de l'historique a importer.
		echo "<h1> Bienvenue sur votre historique ".$_SESSION['client'][1]."</h1>";
	}else{
		include('connexion2.php');//sinon le client doit se connecter
	}


	?>
	
	
       
</body>
</html>