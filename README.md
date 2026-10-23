# Mon_Premier_Mod-le_IA
Mon Premier Modèle de Machine Learning en Python 

from sklearn.linear_model import LinearRegression
import numpy as np

print("=" * 40)
print("   🤖 MON PREMIER MODÈLE D'IA")
print("=" * 40)

# 1. Données d'entraînement (Heures de révision)
X = np.array([[1], [2], [3], [4], [6], [7]]) 

# Les RÉSULTATS correspondants (Notes sur 20)
y = np.array([5, 8, 12, 15, 18, 19])

print("🧠 Entraînement de l'IA en cours...")
modele = LinearRegression()

# 2. Entraînement
modele.fit(X, y)
print("✅ IA entraînée avec succès !")

# 3. Prédiction
heures_etude = np.array([[5]])
prediction = modele.predict(heures_etude)

print("\n🔮 PRÉDICTION DE L'IA :")
print(f"Si un élève révise 5 heures, il aura environ : {prediction[0]:.2f} / 20")

heures_etude_2 = np.array([[8]])
prediction_2 = modele.predict(heures_etude_2)
print(f"Si un élève révise 8 heures, il aura environ : {prediction_2[0]:.2f} / 20")