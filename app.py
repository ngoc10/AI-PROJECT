import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load mô hình đã huấn luyện của bạn
model = tf.keras.models.load_model('my_model.keras')

# Định nghĩa các nhãn
labels = ['desert', 'mountains', 'sea', 'sunset', 'trees']

# Định nghĩa hàm xử lý ảnh và dự đoán
def predict(image):
    # Chuyển đổi ảnh thành tensor
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Dự đoán
    probabilities = model.predict(image_array)[0]

    return dict(zip(labels, probabilities))

# Giao diện Streamlit
st.title("Nhận diện sự vật trong ảnh phong cảnh")

# Upload ảnh
uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã upload.', use_column_width=True)
    st.write("")
    st.write("Đang dự đoán...")

    # Dự đoán
    predictions = predict(image)
    
    # Hiển thị kết quả dưới dạng bảng
    st.write("Kết quả dự đoán:")
    st.table(predictions.items())

    # Tìm 3 nhãn có giá trị cao nhất
    top3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    st.write("3 nhãn có giá trị dự đoán cao nhất:")
    for label, score in top3:
        st.write(f"{label}: {score:.4f}")
