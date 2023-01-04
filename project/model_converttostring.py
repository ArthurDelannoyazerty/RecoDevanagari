def convert_highest_percentage_response_to_string(response_list, percentage_list):
  # Trouver l'index de la réponse avec le pourcentage le plus haut
  highest_percentage_index = percentage_list.index(max(percentage_list))
  
  # Récupérer la réponse correspondante
  highest_percentage_response = response_list[highest_percentage_index]
  
  # Convertir la réponse en chaîne de caractères
  highest_percentage_response_str = str(highest_percentage_response)
  
  return highest_percentage_response_str

# Exemple d'utilisation de la fonction
response_list = ["réponse 1", "réponse 2", "réponse 3"]
percentage_list = [0.5, 0.7, 0.3]
highest_percentage_response_str = convert_highest_percentage_response_to_string(response_list, percentage_list) 
#S'assurer que chaque réponse ait bien un pourcentage associé
print(highest_percentage_response_str) # Affiche "réponse 2"

#ajouter des vérifications sur la liste vide et si tous les pourcentages sont nuls
