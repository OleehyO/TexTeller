import os
import io
import base64
import tempfile
import shutil
import streamlit as st

from PIL import Image
from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller
from utils import to_katex


html_string = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/429-troll/download" width="50">
        TexTeller
        <img src="https://slackmojis.com/emojis/429-troll/download" width="50">
    </h1>
'''

suc_gif_html = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
    </h1>
'''

fail_gif_html = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
    </h1>
'''

tex = r'''
\documentclass{{article}}
\usepackage[
  left=1in,  % 左边距
  right=1in, % 右边距
  top=1in,   % 上边距
  bottom=1in,% 下边距
  paperwidth=40cm,  % 页面宽度
  paperheight=40cm % 页面高度，这里以A4纸为例
]{{geometry}}

\usepackage[utf8]{{inputenc}}
\usepackage{{multirow,multicol,amsmath,amsfonts,amssymb,mathtools,bm,mathrsfs,wasysym,amsbsy,upgreek,mathalfa,stmaryrd,mathrsfs,dsfont,amsthm,amsmath,multirow}}

\begin{{document}}

{formula}

\pagenumbering{{gobble}}
\end{{document}}
'''


@st.cache_resource
def get_model():
    return TexTeller.from_pretrained(os.environ['CHECKPOINT_DIR'])

@st.cache_resource
def get_tokenizer():
    return TexTeller.get_tokenizer(os.environ['TOKENIZER_DIR'])

def get_image_base64(img_file):
    buffered = io.BytesIO()
    img_file.seek(0)
    img = Image.open(img_file)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

model = get_model()
tokenizer = get_tokenizer()

if "start" not in st.session_state:
    st.session_state["start"] = 1
    st.toast('Hooray!', icon='🎉')


#  ============================     pages      =============================== #

st.markdown(html_string, unsafe_allow_html=True)

uploaded_file = st.file_uploader("",type=['jpg', 'png', 'pdf'])

if uploaded_file:
    img = Image.open(uploaded_file)

    temp_dir = tempfile.mkdtemp()
    png_file_path = os.path.join(temp_dir, 'image.png')
    img.save(png_file_path, 'PNG')

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
        max-width: 500px;
        max-height: 500px;
    }}
    </style>
    <div class="centered-container">
        <img src="data:image/png;base64,{img_base64}" class="centered-image" alt="Input image">
        <p style="color:gray;">Input image ({img.height}✖️{img.width})</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    with st.spinner("Predicting..."):
        uploaded_file.seek(0)
        TexTeller_result = inference(
            model,
            tokenizer,
            [png_file_path],
            True if os.environ['USE_CUDA'] == 'True' else False,
            int(os.environ['NUM_BEAM'])
        )[0]
        st.success('Completed!', icon="✅")
        st.markdown(suc_gif_html, unsafe_allow_html=True)
        katex_res = to_katex(TexTeller_result)
        st.text_area(":red[Predicted formula]", katex_res, height=150)
        st.latex(katex_res)

        shutil.rmtree(temp_dir)

#  ============================     pages      =============================== #
