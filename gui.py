import os
import tempfile

import streamlit as st

st.title('Virtual Try-On')
st.write(" ------ ")

# model


SIDEBAR_OPTION_DEMO_IMAGE = "Select Demo Images"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload Images"
SIDEBAR_OPTION_CREDIT = "Credit Page"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_CREDIT]


def main():

    app_mode = st.sidebar.selectbox("Please Select", SIDEBAR_OPTIONS)
    if app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        st.sidebar.write(" ------ ")
        st.sidebar.write("Slect Demo Images")

        # directory = os.path.join()
        photos_person = []
        photos_clothes = []
        ##

        option_person = st.sidebar.selectbox('select a sample image of person', photos_person)
        potion_cloth = st.sidebar.selectbox('slect a sample image of clothes', photos_clothes)

        pressed = st.sidebar.button('Fuse!')

        if pressed:
            st.empty()  # 清空右边
            st.sidebar.write("Fusing...")

            # pic_person = os.path.join(directory, option_person)
            # pic_clothes = os.path.join(directory, option_clothes)
            # run_app(pic_person, pic_clothes)

    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
        st.sidebar.write(" ------ ")
        st.sidebar.write("Upload Images")

        f_p = st.sidebar.file_uploader("Please Select to Upload an Person Image", type=['png', 'jpg'])
        f_c = st.sidebar.file_uploader("Please Select to Upload an Clothes Image", type=['png', 'jpg'])

        if f_p is not None and f_c is not None:
            st.write('ok')
            # tfile_p = tempfile.NamedTemporaryFile(delete=True)
            # tfile_p.write(f_p.read())
            # tfile_c = tempfile.NamedTemporaryFile(delete=True)
            # tfile_c.write(f_c.read())
            # run_app(tfile_p, tfile_c)



    elif app_mode == SIDEBAR_OPTION_CREDIT:

        st.empty()
        st.subheader("The Website Builder")
        first_column = st.columns(1)

        # first_column.write("Olivia Wei")
        st.write("Olivia Wei :sunglasses:")
        expandar_model = st.expander("Model Credit")
        expandar_model.write('''@article{ge2021parser,  
                             title={Parser-Free Virtual Try-on via Distilling Appearance Flows},    
                            author={Ge, Yuying and Song, Yibing and Zhang, Ruimao and Ge, Chongjian and Liu, Wei and Luo, Ping},   
                            journal={arXiv preprint arXiv:2103.04559},
                            year={2021}'''
                            )


main()

expander_faq = st.expander("more info")
expander_faq.write("DNF is real!")