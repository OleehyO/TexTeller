import os
import io
import base64
import tempfile
import time
import subprocess
import shutil
import streamlit as st

from PIL import Image, ImageChops
from pathlib import Path
from pdf2image import convert_from_path
from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller


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

def rendering(formula: str, out_img_path: Path) -> bool:
    build_dir = out_img_path / 'build'
    build_dir.mkdir(exist_ok=True, parents=True)
    f = build_dir / 'formula.tex'
    f.touch(exist_ok=True)
    f.write_text(tex.format(formula=formula))

    p = subprocess.Popen([
        'xelatex', 
        f'-output-directory={build_dir}', 
        '-interaction=nonstopmode',
        '-halt-on-error',
        f'{f}'
    ])
    p.communicate()
    return p.returncode == 0

def pdf_to_pngbytes(pdf_path):
    images = convert_from_path(pdf_path, dpi=400,first_page=1, last_page=1)
    trimmed_images = trim(images[0])
    png_image_bytes = io.BytesIO()
    trimmed_images.save(png_image_bytes, format='PNG')
    png_image_bytes.seek(0)
    return png_image_bytes

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


model = get_model()
tokenizer = get_tokenizer()
# check if xelatex is installed
xelatex_installed = os.system('which xelatex > /dev/null 2>&1') == 0

if "start" not in st.session_state:
    st.session_state["start"] = 1

    if xelatex_installed:
        st.toast('Hooray!', icon='🎉')
        time.sleep(0.5)
        st.toast("Xelatex have been detected.", icon='✅')
    else: 
        st.error('xelatex is not installed. Please install it before using TexTeller.')


#  ============================     pages      =============================== #

st.markdown(html_string, unsafe_allow_html=True)

uploaded_file = st.file_uploader("",type=['jpg', 'png', 'pdf'])

if xelatex_installed:
    st.caption('🥳 Xelatex have been detected, rendered image will be displayed in the web page.')
else:
    st.caption('😭 Xelatex is not detected, please check the resulting latex code by yourself, or check ... to have your xelatex setup ready.')

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
        TeXTeller_result = inference(
            model,
            tokenizer,
            [png_file_path],
            True if os.environ['USE_CUDA'] == 'True' else False,
            int(os.environ['NUM_BEAM'])
        )[0]
        if not xelatex_installed:
            st.markdown(fail_gif_html, unsafe_allow_html=True)
            st.warning('Unable to find xelatex to render image. Please check the prediction results yourself.', icon="🤡")
            txt = st.text_area(
                ":red[Predicted formula]",
                TeXTeller_result,
                height=150,
            )
        else:
            is_successed = rendering(TeXTeller_result, Path(temp_dir))
            if is_successed:
                # st.code(TeXTeller_result, language='latex')

                img_base64 = get_image_base64(pdf_to_pngbytes(Path(temp_dir) / 'build' / 'formula.pdf'))
                st.markdown(suc_gif_html, unsafe_allow_html=True)
                st.success('Successfully rendered!', icon="✅")
                txt = st.text_area(
                    ":red[Predicted formula]",
                    TeXTeller_result,
                    height=150,
                )
                # st.latex(TeXTeller_result)
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
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(fail_gif_html, unsafe_allow_html=True)
                st.error('Rendering failed. You can try using a higher resolution image or splitting the multi line formula into a single line for better results.', icon="❌")
                txt = st.text_area(
                    ":red[Predicted formula]",
                    TeXTeller_result,
                    height=150,
                )

        shutil.rmtree(temp_dir)

#  ============================     pages      =============================== #
