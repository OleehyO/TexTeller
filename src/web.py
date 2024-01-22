import streamlit as st
import io
import base64
import requests

from PIL import Image

def post_image(server_url, img_rb):
    response = requests.post(server_url, files={'image': img_rb})
    return response.text


#  ============================     pages      =============================== #
# 使用 Markdown 和 HTML 将标题居中
# with st.columns(3)[1]:
#     st.title(":rainbow[TexTeller] :sparkles:")

# HTML字符串，包含内联CSS用于彩色和居中
# html_string = """
#     <h1 style="color: orange; text-align: center;">
#         ✨ TexTeller ✨
#     </h1>
# """
html_string = """
    <h1 style="color: orange; text-align: center;">
        🔥👁️ OCR
    </h1>
"""
st.markdown(html_string, unsafe_allow_html=True)



if "start" not in st.session_state:
    st.balloons()
    st.session_state["start"] = 1

# 上传图片
uploaded_file = st.file_uploader("",type=['jpg', 'png'])

# 显示上传图片
if uploaded_file:
    # 打开上传图片
    img = Image.open(uploaded_file)
    # st.image(uploaded_file, caption=f"Input image ({img.height}✖️{img.width})")

    # 将 BytesIO 对象转换为 Base64 编码
    def get_image_base64(img_file):
        buffered = io.BytesIO()
        img_file.seek(0)  # 重置文件指针位置
        img = Image.open(img_file)
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    img_base64 = get_image_base64(uploaded_file)

    # 使用Markdown和HTML创建一个居中的图片容器
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

    # 预测
    with st.spinner("Predicting..."):
        # 预测结果
        server_url = 'http://localhost:8000/'
        uploaded_file.seek(0)
        TeXTeller_result = post_image(server_url, uploaded_file)
        TeXTeller_result = r"\begin{align*}" + '\n' + TeXTeller_result + '\n' + r'\end{align*}'
        # tab1, tab2 = st.tabs(["✨TeXTeller✨", "pix2tex:gray[(9.6K⭐)️]"])
        tab1, tab2 = st.tabs(["🔥👁️", "pix2tex:gray[(9.6K⭐)️]"])
        # with st.container(border=True):
        with tab1:
            st.latex(TeXTeller_result)
            st.write("")
            st.code(TeXTeller_result, language='latex')
            st.success('Done!')

#  ============================     pages      =============================== #