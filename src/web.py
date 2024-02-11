import os
import io
import base64
import tempfile
import streamlit as st

from PIL import Image
from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller


@st.cache_resource
def get_model():
    return TexTeller.from_pretrained(os.environ['CHECKPOINT_DIR'])


@st.cache_resource
def get_tokenizer():
    return TexTeller.get_tokenizer(os.environ['TOKENIZER_DIR'])


model = get_model()
tokenizer = get_tokenizer()


#  ============================     pages      =============================== #
html_string = '''
    <h1 style="color: orange; text-align: center;">
        ✨ TexTeller ✨
    </h1>
'''
st.markdown(html_string, unsafe_allow_html=True)

if "start" not in st.session_state:
    st.balloons()
    st.session_state["start"] = 1

uploaded_file = st.file_uploader("",type=['jpg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)

    temp_dir = tempfile.mkdtemp()
    png_file_path = os.path.join(temp_dir, 'image.png')
    img.save(png_file_path, 'PNG')

    def get_image_base64(img_file):
        buffered = io.BytesIO()
        img_file.seek(0)
        img = Image.open(img_file)
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    img_base64 = get_image_base64(uploaded_file)

    st.markdown(f"""
    <style>
    .centered-container {{
        text-align: center;
    }}
    .centered-image {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 700px;
    }}
    </style>
    <div class="centered-container">
        <img src="data:image/png;base64,{img_base64}" class="centered-image" alt="Input image">
        <p style="color:gray;">Input image ({img.height}✖️{img.width})</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    with st.spinner("Predicting..."):
        uploaded_file.seek(0)
        TeXTeller_result = inference(
            model,
            tokenizer,
            [png_file_path],
            True if os.environ['USE_CUDA'] == 'True' else False,
            int(os.environ['NUM_BEAM'])
        )[0]

        # st.subheader(':rainbow[Predict] :sunglasses:', divider='rainbow')
        st.subheader(':sunglasses:', divider='gray')
        st.latex(TeXTeller_result)
        st.code(TeXTeller_result, language='latex')
        st.success('Done!')

#  ============================     pages      =============================== #
