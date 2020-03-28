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
	<!--<h1> Aidez-nous à évoluer activement</h1>-->

    <!--BLOCK BLEU-->
    <!-- <div id="blocklabel">
        <p>
            Afin d’améliorer les performances du site Pixoglyphe et de vous proposer le contenu le plus complet, nous vous proposons de labelliser les images pour lequel nous avons besoin d’aide. C’est grâce à vous que l’outil pourra se développer et s’avérer être de plus en plus efficace. Pour assurer la fiabilité des informations, la labellisation est réservée aux experts et se fait grâce aux code utilisés sur le site Karnak.             
        </p>
        <p><strong>Pensez à labelliser uniquement les signes dont vous êtes certain du code.</strong></p>
    </div> -->

    <!--TITRE ANALYSE-->
	<!-- <h2>Je labellise des images </h2> -->
	
	<?php

	session_start();
	
	if(isset($_SESSION['client'])){//si la session client est ouverte
		//page de l'historique a importer.
		echo "<h1> Bienvenue sur l'espace de labélisation ".$_SESSION['client'][1]."</h1>";
		
		echo"<h1> Aidez-nous à évoluer activement</h1>";
		//BLOCK BLEU
		echo '<div id="blocklabel">';
        echo'<p>Afin d’améliorer les performances du site Pixoglyphe et de vous proposer le contenu le plus complet, 
		nous vous proposons de labelliser les images pour lequel nous avons besoin d’aide. 
		C’est grâce à vous que l’outil pourra se développer et s’avérer être de plus en plus efficace. 
		Pour assurer la fiabilité des informations, 
		la labellisation est réservée aux experts et se fait grâce aux code utilisés sur le site Karnak.</p>';
        echo'<p><strong>Pensez à labelliser uniquement les signes dont vous êtes certain du code.</strong></p>';
		echo'</div>';

		//TITRE ANALYSE
		echo'<h2>Je labellise des images </h2>';
		
		
	}else{
		include('connexion2.php');//sinon le client doit se connecter
	}


	?>
	
	
	
	
	
	
       
</body>
</html>