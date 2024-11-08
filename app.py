import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
import json

# --- Configuración del modelo ------------------------------------------
# Definir el modelo (utilizando MobileNetV2 como ejemplo)
model = models.mobilenet_v2(pretrained=False)  # Usamos 'pretrained=False' porque vamos a cargar los pesos propios
num_classes = len(os.listdir('Famous_2'))  # Obtener el número de clases
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)  # Modificar la última capa del modelo para las clases de tu dataset

# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load('modelo_entrenado.pth'))  # Cargar los pesos entrenados en el modelo

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Poner el modelo en modo evaluación
model.eval()

# Cargar el diccionario de los famosos desde el archivo JSON
with open('famous_dict.json', 'r') as f:
    famous_dict = json.load(f)

# --- Frontend de la aplicación ---------------------------------------
st.title("¿Quién es el famoso?")  # Título principal

# --- Subir imagen -----------------------------------------------------
uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocesar la imagen
    img = Image.open(uploaded_file).convert('RGB')  # Asegurarse de que la imagen sea RGB
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # Cambiar el tamaño
        transforms.ToTensor(),           # Convertir la imagen a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # Añadir una dimensión extra para el batch y mover a dispositivo

    # --- Hacer la predicción ------------------------------------------
    with torch.no_grad():  # No necesitamos calcular gradientes
        outputs = model(img_tensor)  # Hacer la predicción
        _, predicted_class = torch.max(outputs, 1)  # Obtener la clase predicha

    # --- Mostrar resultados -----------------------------------------
    # Mostrar la imagen subida
    st.image(img, caption='Imagen subida', use_column_width=True)

    # Obtener el nombre del famoso desde el diccionario
    famous_name = list(famous_dict.keys())[predicted_class.item()]
    st.write(f"Te pareces a: **{famous_name}**")

    # Mostrar el porcentaje de similitud
    confidence = torch.nn.functional.softmax(outputs, dim=1)  # Obtener la probabilidad de la predicción
    similarity_percentage = confidence[0][predicted_class.item()] * 100
    st.write(f"Similitud: **{similarity_percentage:.2f}%**")

    # --- Mostrar la imagen del famoso correspondiente ------------------
    famous_image_path = famous_dict.get(famous_name, None)
    if famous_image_path:
        famous_img = Image.open(famous_image_path)
        st.image(famous_img, caption=f"Te pareces a {famous_name}", use_column_width=True)
    else:
        st.write("No se encontró la imagen del famoso.")

    # --- Estilización adicional --------------------------------------
    st.markdown("""
    <style>
    .title {
        font-size: 30px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .image-caption {
        text-align: center;
        font-size: 18px;
        font-style: italic;
    }
    .result-text {
        font-size: 24px;
        color: #FF5722;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Usar las clases CSS para personalizar el texto
    st.markdown('<div class="title">¿Quién es el famoso?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-text">Te pareces a: {famous_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-text">Similitud: {similarity_percentage:.2f}%</div>', unsafe_allow_html=True)
