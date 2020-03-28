
<!DOCTYPE html>
<html>

<head>

<title> Deconnexion</title>
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8"/>

</head>


<body>


<?php
session_start();
session_destroy();
echo "<meta http-equiv='refresh' content='2;URL=accueil.php'>";

?>

<!-- Votre code HTML /PHP -->
<h1 class='messagePage' > Vous êtes déconnecté </h1>
</br>
<a href="index.php"> Retour au menu </a>
<!-- Votre code HTML /PHP -->

</body>
</html>