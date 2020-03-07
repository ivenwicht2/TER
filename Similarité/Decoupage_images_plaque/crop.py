from PIL import Image

# Définition de la fonction de découpage

def crop(image_path, coords, saved_location):
    """
    @param image_path: Chemin vers l'image à découper
    @param coords: Tuple de coordonnées x/y (x1, y1, x2, y2)
    @param saved_location: Chemin ou il faut enregistrer l'image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()

# Appel à la fonction 

if __name__ == '__main__':
    image = '/Users/priscille/Desktop/Decoupage_images_plaque/69166.jpg'
    coord = ((1156, 372, 1421, 807),(1640, 799, 2117, 1022),(2179, 803, 2419, 1259))
    for i in range(len(coord)) :
        crop(image,coord[i],'/Users/priscille/Desktop/Decoupage_images_plaque/69166cropped'+str(i)+'.jpg')