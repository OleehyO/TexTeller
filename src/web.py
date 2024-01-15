import streamlit as st
import time

from stqdm import stqdm

# 使用 Markdown 和 HTML 将标题居中
with st.columns(3)[1]:
    st.title(":rainbow[TexTeller] :sparkles:")

if "start" not in st.session_state:
    st.balloons()
    st.session_state["start"] = 1

uploaded_file = st.file_uploader("",type=['jpg', 'png'])
st.divider()

if uploaded_file:
    st.image(uploaded_file, caption="Input image")

for _ in stqdm(range(10), st_container=st.sidebar):
    time.sleep(0.1)

with st.spinner('Wait for it...'):
    time.sleep(5)

st.success('Done!')


with st.empty():
    for seconds in range(60):
        st.write(f"⏳ {seconds} seconds have passed")
        time.sleep(1)
    st.write("✔️ 1 minute over!")