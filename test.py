import tempfile
import imghdr
import requests
import shutil
from PIL import Image
from pathlib import Path
# import tensorflow as tf
from data.base_dataset import BaseDataset, get_params, get_transform

import streamlit as st
import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F

device = torch.device("cpu")

st.title('Virtual Try-On')
st.write(" ------ ")

# model
DEFAULT_MODEL_BASE_DIR = 'models'
DEFAULT_DATA_BASE_DIR = 'dataset'
IMAGE_DIR_IMG = 'test_img'
IMAGE_DIR_CLOTHES = 'test_clothes'
IMAGE_DIR_EDGE = 'test_edge'


MODEL_W_WEIGHTS = f'{DEFAULT_MODEL_BASE_DIR}/warp_model_final.pth'
MODEL_G_WEIGHTS = f'{DEFAULT_MODEL_BASE_DIR}/gen_model_final.pth'

MODEL_WEIGHTS_G_DEPLOYMENT_URL = 'https://github.com/HelpMeCodePls/temp/releases/download/test/gen_model_final.pth'
MODEL_WEIGHTS_W_DEPLOYMENT_URL = 'https://github.com/HelpMeCodePls/temp/releases/download/test/warp_model_final.pth'



SIDEBAR_OPTION_DEMO_IMAGE = "Select Demo Images"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload Images"
SIDEBAR_OPTION_CREDIT = "Credit Page"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_CREDIT]


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename

# Modified from https://github.com/thoppe/streamlit-skyAR/blob/master/streamlit_app.py

def load_modal():
    download_file(url=MODEL_WEIGHTS_W_DEPLOYMENT_URL, local_filename=MODEL_W_WEIGHTS)
    download_file(url=MODEL_WEIGHTS_G_DEPLOYMENT_URL, local_filename=MODEL_G_WEIGHTS)



def run_app(p_path, c_path, e_path):

    opt = TestOptions().parse()
    opt.warp_checkpoint = MODEL_W_WEIGHTS
    opt.gen_ceckpoint = MODEL_G_WEIGHTS

    start_epoch, epoch_iter = 1, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(dataset_size)
    dataset_size = 1

    # fine_height = 256
    # fine_weight = 192

    warp_model = AFWM(opt, 3)
    warp_model.eval()
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    gen_model.eval()
    load_checkpoint(gen_model, opt.gen_checkpoint)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size / opt.batchSize


    # P = Image.open(p_path).convert('RGB')
    # params = get_params(opt, P.size)
    # transform = get_transform(opt, params)
    # transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    #
    # P_tensor = transform(P)
    #
    # C = Image.open(c_path).convert('RGB')
    # C_tensor = transform(C)
    #
    # E = Image.open(e_path).convert('L')
    # E_tensor = transform_E(E)
    #
    # input_dict = {'image': P_tensor, 'clothes': C_tensor, 'edge': E_tensor}
    #


    for epoch in range(1,2):

        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # real_image = P_tensor
            # clothes = C_tensor
            # ##edge is extracted from the clothes image with the built-in function in python
            # edge = E_tensor
            real_image = data['image']
            clothes = data['clothes']
            edge = data['edge']


            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
            clothes = clothes * edge

            flow_out = warp_model(real_image.to(device), clothes.to(device))
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                              mode='bilinear', padding_mode='zeros')

            gen_inputs = torch.cat([real_image.to(device), warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            # path = 'results/' + opt.name
            # os.makedirs(path, exist_ok=True)
            # sub_path = path + '/PFAFN'
            # os.makedirs(sub_path,exist_ok=True)

            ## 显示 ##
            # i_column, c_column = st.columns(2)
            i_column = st.columns(1)
            a = real_image.float().to(device)
            b = clothes.to(device)
            c = p_tryon
            combine = torch.cat([a[0], b[0], c[0]], 2).squeeze()
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            rgb = (cv_img * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            st.image(bgr, caption="Demo Result")

            # if step % 1 == 0:
            #     a = real_image.float().cuda()
            #     b= clothes.cuda()
            #     c = p_tryon
            #     combine = torch.cat([a[0],b[0],c[0]], 2).squeeze()
            #     cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            #     rgb=(cv_img*255).astype(np.uint8)
            #     bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)

            step += 1
            if epoch_iter >= dataset_size:
                break


# @st.cache
# def ensure_model_exists():
#
#     save_dest = Path(DEFAULT_MODEL_BASE_DIR)
#     save_dest.mkdir(exist_ok=True)
#
#     f_checkpoint_W = Path(MODEL_W_WEIGHTS)
#     f_checkpoint_G = Path(MODEL_G_WEIGHTS)
#
#     if not f_checkpoint_W.exists():
#         with st.spinner("Downloading model weights... this may take up to a few minutes. (~150 MB) Please don't interrupt it."):
#             download_file(url=MODEL_WEIGHTS_W_DEPLOYMENT_URL, local_filename=MODEL_W_WEIGHTS)
#
#     if not f_checkpoint_G.exists():
#         with st.spinner("Downloading model weights... this may take up to a few minutes. (~150 MB) Please don't interrupt it."):
#             download_file(url=MODEL_WEIGHTS_G_DEPLOYMENT_URL, local_filename=MODEL_G_WEIGHTS)
#
#     # f_architecture = Path(MODEL_JSON)
#
#     # if not f_architecture.exists():
#     #     with st.spinner("Downloading model architecture... this may take a few seconds. Please don't interrupt it."):
#     #         download_file(url=MODEL_JSON_DEPLOYMENT_URL, local_filename=MODEL_JSON)
#
#     # return AppHelper(model_weights=MODEL_WEIGHTS, model_json=MODEL_JSON)
#     return AppHelper(model_w_weights=MODEL_W_WEIGHTS, model_g_weights=MODEL_G_WEIGHTS)

#         rescale_f = cv2.imread(img)
#         rescale_f = cv2.cvtColor(rescale_f,cv2.COLOR_BGR2RGB)
#         rescale_f = cv2.resize(rescale_f, dsize=(256,256))


# @st.cache(allow_output_mutation=True, hash_funcs=HASH_FUNCS)


# def load_model():
#     config = tf.ConfigProto(allow_soft_placement=True)
#     session = tf.Session(config=config)
#     with session.as_default():
#         handle = ensure_model_exists()
#
#         # Needed to ensure model can be cached and called
#         handle.model._make_predict_function()
#         handle.model.summary()
#
#     return handle, session

def main():

    app_mode = st.sidebar.selectbox("Please Select", SIDEBAR_OPTIONS)
    if app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        st.sidebar.write(" ------ ")
        st.sidebar.write("Slect Demo Images")

        directory_img = os.path.join(DEFAULT_DATA_BASE_DIR, IMAGE_DIR_IMG)
        directory_clo = os.path.join(DEFAULT_DATA_BASE_DIR, IMAGE_DIR_CLOTHES)
        directory_edge = os.path.join(DEFAULT_DATA_BASE_DIR, IMAGE_DIR_EDGE)

        photos_person = []
        photos_clothes = []
        photos_edges = []


        for file in os.listdir(directory_img):
            filepath = os.path.join(directory_img, file)

            # Find all valid images
            if imghdr.what(filepath) is not None:
                photos_person.append(file)

            # edge 和 clo同名倒是
        for file in os.listdir(directory_clo):
            filepath = os.path.join(directory_clo, file)

            # Find all valid images
            if imghdr.what(filepath) is not None:
                photos_clothes.append(file)

        # for file in os.listdir(directory_edge):
        #     filepath = os.path.join(directory_edge, file)
        #
        #     # Find all valid images
        #     if imghdr.what(filepath) is not None:
        #         photos_edges.append(file)

        photos_person.sort()
        photos_clothes.sort()
        # photos_edges.sort()

        option_person = st.sidebar.selectbox('select a sample image of person', photos_person)
        option_clothes = st.sidebar.selectbox('slect a sample image of clothes', photos_clothes)

        pressed = st.sidebar.button('Fuse!')

        if pressed:
            st.empty()  # 清空右边
            st.sidebar.write("Fusing...")

            pic_person = os.path.join(directory_img, option_person)
            pic_clothes = os.path.join(directory_clo, option_clothes)
            pic_edge = os.path.join(directory_edge, option_clothes)
            run_app(pic_person, pic_clothes, pic_edge)

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
