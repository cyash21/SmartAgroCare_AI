from tensorflow.keras.models import load_model

model = load_model(r"C:/Users/Durgesh Nikam/Downloads/PROJECT/AI-Based-Plant-Health-monitoring-System/Models/Disease/plant_disease_model.keras")

model.save("plant_disease_model.h5")

print("Conversion Done! Saved as plant_disease_model.h5")
