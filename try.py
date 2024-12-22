import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import zipfile

# Fungsi untuk memuat model
@st.cache_resource
def load_inception_model():
    model = load_model('googlenet_covid_pneumonia1.keras')
    return model

model = load_inception_model()

# Fungsi untuk klasifikasi gambar
def classify_image(image, model):
    # Ubah ukuran gambar sesuai dengan input model
    img = image.resize((299, 299))
    img = img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension

    # Prediksi dengan model
    pred = model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]
    categories = ['COVID', 'NORMAL', 'PNEUMONIA']
    label = categories[class_idx]
    confidence = np.max(pred)
    return label, confidence

# Fungsi untuk membuat PDF dengan gambar
def create_pdf_with_images(results):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica", 12)
    y_position = height - 40  # Mulai dari bagian atas halaman

    c.drawString(40, y_position, "Hasil Klasifikasi Gambar Paru-Paru")
    y_position -= 20
    c.drawString(40, y_position, "----------------------------------")
    y_position -= 20
    
    for file_name, label, confidence, image in results:
        # Tampilkan hasil klasifikasi
        c.drawString(40, y_position, f"{file_name}: {label} ({confidence * 100:.2f}%)")
        y_position -= 20
        
        # Tambahkan gambar ke PDF
        img = ImageReader(image)
        if y_position < 240:  # Jika ruang tidak cukup, buat halaman baru
            c.showPage()
            y_position = height - 40
        c.drawImage(img, 40, y_position - 200, width=200, height=200)
        y_position -= 220

        if y_position < 40:  # Jika sudah mencapai bagian bawah halaman, buat halaman baru
            c.showPage()
            y_position = height - 40

    # Selesai menulis ke buffer
    c.save()
    buffer.seek(0)
    return buffer

# Fungsi untuk memproses file ZIP
def process_zip_file(zip_file):
    results = []
    with zipfile.ZipFile(zip_file) as z:
        for file_name in z.namelist():
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                with z.open(file_name) as f:
                    image = Image.open(f).convert('RGB')
                    label, confidence = classify_image(image, model)
                    results.append((file_name, label, confidence, image))
    return results

# Navigasi sidebar
menu = ["Beranda", "Klasifikasi", "Tentang"]
choice = st.sidebar.selectbox("Navigasi", menu)

if choice == "Beranda":
    # Menampilkan judul dan deskripsi
    st.markdown("""
    <h1 style="text-align: center;">ALus</h1>
    <h2 style="text-align: center;">Artificial Lungs Disease Detection</h2>
    <p style="text-align: center;">Aplikasi klasifikasi gambar paru-paru berdasarkan model InceptionV3.</p>
    """, unsafe_allow_html=True)

    # Menampilkan gambar di bawah deskripsi
    st.image("Lung.jpg", use_column_width=True)

    # Teks justify
    st.markdown("""
    <p style="text-align: justify; font-size: 18px;">
        Kesehatan paru-paru sangat penting untuk kualitas hidup yang optimal. Penyakit paru seperti pneumonia dan infeksi akibat COVID-19 dapat
        mempengaruhi fungsi paru dan menyebabkan gangguan pernapasan. Dengan kemajuan teknologi, deteksi dini melalui gambar X-ray
        dapat membantu mendiagnosis penyakit paru lebih cepat dan lebih akurat.
    </p>
    <p style="text-align: justify; font-size: 18px;">
        COVID-19, yang disebabkan oleh virus SARS-CoV-2, dapat menular melalui udara dan menyebabkan gangguan serius pada paru-paru.
        Pneumonia adalah infeksi yang menyebabkan peradangan pada kantung udara di paru-paru, dan sering kali terkait dengan komplikasi
        dari infeksi seperti COVID-19.
    </p>
    <p style="text-align: justify; font-size: 18px;">
        ALus bertujuan untuk memberikan alat bantu dalam deteksi penyakit paru berdasarkan gambar X-ray, sehingga mempermudah diagnosis
        dan perawatan lebih cepat.
    </p>
    """, unsafe_allow_html=True)

elif choice == "Klasifikasi":
    st.title("Klasifikasi Gambar Paru-Paru")
    option = st.radio("Pilih metode unggah:", ["Unggah beberapa gambar", "Unggah file ZIP"])

    results = []
    if option == "Unggah beberapa gambar":
        uploaded_files = st.file_uploader(
            "Unggah satu atau beberapa gambar",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write("### Hasil Klasifikasi")
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)

                label, confidence = classify_image(image, model)
                st.write(f"**{uploaded_file.name}**: {label} ({confidence * 100:.2f}%)")
                results.append((uploaded_file.name, label, confidence, image))

    elif option == "Unggah file ZIP":
        zip_file = st.file_uploader("Unggah file ZIP berisi gambar", type=["zip"])
        if zip_file:
            st.write("### Hasil Klasifikasi")
            results = process_zip_file(zip_file)
            for file_name, label, confidence, image in results:
                st.image(image, caption=f"Gambar: {file_name}", use_column_width=True)
                st.write(f"**{file_name}**: {label} ({confidence * 100:.2f}%)")

    # Menambahkan tombol untuk mengunduh PDF hasil
    if results:
        pdf_buffer = create_pdf_with_images(results)
        st.download_button(
            label="Unduh Hasil Klasifikasi (PDF)",
            data=pdf_buffer,
            file_name="hasil_klasifikasi.pdf",
            mime="application/pdf"
        )

elif choice == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("Aplikasi ini menggunakan model deep learning (InceptionV3) untuk mengklasifikasikan gambar X-ray paru-paru menjadi tiga kategori: COVID, Pneumonia, dan Normal.")
    st.write("### Pengembang:")
    st.markdown("- Wildan Miladji")
    st.markdown("- Robert William")
    st.markdown("- Rayhan Gading")
    st.markdown("- Yahya Bachtiar")
    st.markdown("- Siti Arwiyah")
    st.write("### Fitur Utama:")
    st.markdown("- **Klasifikasi gambar individu**: Unggah satu atau beberapa gambar X-ray untuk klasifikasi.")
    st.markdown("- **Klasifikasi file ZIP**: Unggah file ZIP berisi banyak gambar untuk klasifikasi.")
    st.markdown("- **Unduh hasil dalam PDF**: Hasil klasifikasi gambar dapat diunduh dalam bentuk PDF.")
    st.markdown("- **Deep Learning Model**: Menggunakan model InceptionV3 yang sudah dilatih.")
