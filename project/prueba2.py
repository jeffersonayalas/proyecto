import streamlit as st
from clarifai.client.model import Model
from PIL import Image
import io

st.title("Aplicación de Análisis de Imágenes")

uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Convertir la imagen a formato bytes
    image_bytes = io.BytesIO(uploaded_file.getvalue())

    model_url = "https://clarifai.com/clarifai/main/models/general-image-detection"
    detector_model = Model(url=model_url, pat="95fbcab28c7a4517a0390af2e60903eb")

    # Leer los bytes de la imagen asegurando que están en el formato correcto (por ejemplo, RGB)
    image = Image.open(image_bytes)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")

    # Obtener predicciones para la imagen cargada
    prediction_response = detector_model.predict_by_bytes(image_bytes.getvalue(), input_type="image")

    regions = prediction_response.outputs[0].data.regions

    for region in regions:
        top_row = round(region.region_info.bounding_box.top_row, 3)
        left_col = round(region.region_info.bounding_box.left_col, 3)
        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
        right_col = round(region.region_info.bounding_box.right_col, 3)

        for concept in region.data.concepts:
            name = concept.name
            value = round(concept.value, 4)
            st.title("Result: ")

            st.write(f"{name}: {value*100}% ")
