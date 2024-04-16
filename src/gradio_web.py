import os

import gradio as gr
from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller
from utils import to_katex
from pathlib import Path


# model     = TexTeller.from_pretrained(os.environ['CHECKPOINT_DIR'])
# tokenizer = TexTeller.get_tokenizer(os.environ['TOKENIZER_DIR'])


css = """
<style>
    .container {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-family: 'Arial';
    }
    .container img {
        height: auto;
    }
    .text {
        margin: 0 15px;
    }
    h1 {
        text-align: center;
        font-size: 50px !important;
    }

    .markdown-style {
        color: #333;  /* 调整颜色 */
        line-height: 1.6;  /* 行间距 */
        font-size: 50px;
    }
    .markdown-style h1, .markdown-style h2, .markdown-style h3 {
        color: #007BFF;  /* 为标题元素指定颜色 */
    }
    .markdown-style p {
        margin-bottom: 1em;  /* 段落间距 */
    }
</style>
"""

theme=gr.themes.Default(),

def fn(img):
    return img

with gr.Blocks(
    theme=theme,
    css=css
) as demo:
    gr.HTML(f'''
    {css}
    <div class="container">
        <img src="https://github.com/OleehyO/TexTeller/raw/main/assets/fire.svg" width="100">
        <h1> 𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛 </h1>
        <img src="https://github.com/OleehyO/TexTeller/raw/main/assets/fire.svg" width="100">
    </div>
    ''')

    with gr.Row(equal_height=True):
        input_img = gr.Image(type="pil", label="Input Image")
        latex_img = gr.Image(label="Predicted Latex", show_label=False)
        input_img.upload(fn, input_img, latex_img)
    
    gr.Markdown(r'$$\fcxrac{7}{10349}$$')
    gr.Markdown('fooooooooooooooooooooooooooooo')


demo.launch()
