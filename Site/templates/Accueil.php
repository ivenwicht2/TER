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
    <img src="barre.png" width="5%">

<!--
    <svg height = "30px" width = "100%">
        <rect width="100%" height="30" x="0" y="0" stroke="#1C1C1C" stroke-width="3" fill="#1C1C1C" />;
        <rect width="25%" height="30" x="0%" y="0" stroke="rgb(231, 231, 231)" stroke-width="3" fill="rgb(231, 231, 231)" />;

-->
        <!--<polygon points="200 30, 210 0 ,400 0, 410 30  " stroke="rgb(231, 231, 231)" stroke-width="3" fill="rgb(231, 231, 231)"/>;
        <polygon points="500 30, 510 0 ,700 0, 710 30  " stroke="#1C1C1C" stroke-width="3" fill="#1C1C1C" />;-->
<!--
        <text x="10%" y="25" fill="black" font-size="18">Accueil</text>;
        <text x="30%" y="25" fill="white" font-size="18">Labellisation</text>
    </svg> 

-->

    <!--TITRE-->
    <h1> Bienvenue sur le site Pixoglyphe</h1>

    <!--BLOCK JAUNE-->
    <div id="blockaccueil">
        <p>
            Le site <i>Pixoglyphe</i> vous permet d'analyser des scènes contenant différents signes hiéroglyphiques. Pour cela, il vous suffit d'importer une image et de lancer l'analyse. L'analyse peut durer quelques minutes et vous permettra de visualiser les différents signes présents sur la scène. Il vous suffira alors de cliquer sur un signe pour obtenir les images similaires. N'oubliez pas, vous pouvez nous aider activement en vous rendant dans l'onglet "Labellisation". 
        </p>
    </div>

    <!--TITRE ANALYSE-->
    <h2>J'analyse un signe </h2>
    
    
    <!--TELECHARGEMENT DE L IMAGE-->
    <div>
        <form method=post enctype=multipart/form-data>
            <input type=file name=photo>
            <input type=submit value=Upload>
        </form>
    </div>
       
</body>
</html>