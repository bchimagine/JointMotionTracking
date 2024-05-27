import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import sys
import random

from Diffeo_losses import NCC, MSE, Grad
from Diffeo_networks import DiffeoDense  
from SitkDataSet import SitkDataset as SData
from tools import ReadFiles as rd
from functools import partial
import utils
import losses
import custom_image3d as ci3d
import rxfm_net
import SimpleITK as sitk 
import time
from pytorch3d import transforms as pt3d_xfms




'''Read parameters by yaml'''
para = rd.read_yaml('./parameters.yml')

''' Load data by json'''
json_file = './data_norm_squ.json'
json_file_val = './data_norm_squ_val.json'
batch_size = para.solver.batch_size
dataset = SData(json_file, "train")

dataset_val = SData(json_file_val, "val")

'''Set device (GPU or CPU)'''
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)

'''Create a DataLoader'''
trainloader = DataLoader(dataset, batch_size= batch_size, shuffle=True)
valloader = DataLoader(dataset_val, batch_size= 1, shuffle=False)



batch_size = para.solver.batch_size
IMG_SIZE = [96,96,96]
loss_func_name = "xfm_6D"

n_conv_chan = 1
n_chan = 64
overfit = True 
running_loss = 0 
def_weight = para.solver.def_weight
net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)
print (net_obj)



if loss_func_name == "xfm_MSE":
    loss_func = partial( losses.xfm_loss_MSE, weight_R=1.0, weight_T=5.0)
elif loss_func_name == "xfm_6D":
    loss_func = partial( losses.xfm_loss_6D, weight_R=1.0, weight_T=5.0)
else:
    print("Loss function not recognized")
    exit(1)

net_obj = net_obj.to(dev)
LR = 0.000025
optimizer = torch.optim.Adam(net_obj.parameters(), lr=LR)

if (para.model.deformable == True):
    Diffeo_net = DiffeoDense(inshape=(IMG_SIZE[1], IMG_SIZE[1], IMG_SIZE[2]),
                      nb_unet_features=[[16, 32, 32], [32, 32, 32, 16, 16]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    diff_net = Diffeo_net.to(dev)
    diff_net = diff_net.to(dev)
    joint_params = list(net_obj.parameters()) + list(diff_net.parameters())
    optimizer = torch.optim.Adam(joint_params, lr=LR)
    criterion = nn.MSELoss()
low = 1
high = 20
low_val = 1
high_val = 5
trans_arr = np.zeros(len(valloader))
angular_arr =  np.zeros(len(valloader))
# '''Training and validation'''        
for epoch in range(para.solver.epochs):
    total= 0; 
    total_val =0; 
    print('epoch:', epoch)
    for idx, image_data in enumerate(trainloader):
        source, temp=image_data
        b = source.shape[0]
        source = source.to(dev).float() 
        # target =target.to(dev).float() 
      
        # print (source.shape)
        optimizer.zero_grad()   

        rx_train = random.randint(low, high)
        ry_train = random.randint(low, high)
        rz_train = random.randint(low, high)

        tx_train = random.randint(low, high)
        ty_train = random.randint(low, high)
        tz_train = random.randint(low, high)
        print ("epoch: ", epoch, "iter: ", idx, "rotation x:", rx_train, "rotation y:", ry_train, "rotation z:", rz_train, "translation x:", tx_train, "translation y:", ty_train, "translation z:", tz_train)
        mat = ci3d.create_transform(
            rx=rx_train, ry=ry_train, rz=rz_train,
            tx=2.0*tx_train/IMG_SIZE[0], ty=2.0*ty_train/IMG_SIZE[1], tz=2.0*tz_train/IMG_SIZE[2]
        )

        mat = mat[np.newaxis,:,:]
        mat = mat[:,0:3,:]
        mat = torch.tensor(mat).float()


        grids = torch.nn.functional.affine_grid(mat, [1,1] + IMG_SIZE).to(dev)
        target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=False)
        # # Start the timer
        # start_time = time.time()
        xfm_1to2 = net_obj.forward((source,target))
        # Stop the timer
        # end_time = time.time()
        # # Calculate the elapsed time
        # elapsed_time = end_time - start_time

        
        loss_val = loss_func(mat.to(dev), xfm_1to2 )

        predicted_grids = torch.nn.functional.affine_grid(xfm_1to2, [1,1] + IMG_SIZE)
        x_aligned = F.grid_sample(source,
                                  grid=predicted_grids,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)
        # loss_val = NCC().loss(target, x_aligned)
        if (para.model.deformable == True):
            loss_val = NCC().loss(target, x_aligned)
            disp, deformed = diff_net(x_aligned, target, registration = True)
            Reg = Grad( penalty= 'l2')
            loss_reg = Reg.loss(disp)
            loss_dist_deform = NCC().loss(target, deformed)
            loss_val = loss_val + def_weight*loss_dist_deform + def_weight*loss_reg

        if (abs(loss_val) > 10e-5):
            loss_val.backward(retain_graph=True)
        optimizer.step()
    

        
        running_loss += loss_val.item()
        total += running_loss
        
        # print ("epoch: ", epoch, " batch loss: ", running_loss)
        running_loss = 0.0

        # rot_real = mat[:,0:3,0:3].detach().cpu().numpy()
        # trans_real = mat[:,0:3,3:].detach().cpu().numpy()
        
        # rot_approx = xfm_1to2[:,0:3,0:3].detach().cpu().numpy()
        # trans_approx = xfm_1to2[:,0:3,3:].detach().cpu().numpy()
        
        # angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="ZYX")
        # angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="ZYX")
        
        # angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
        # translation_error = np.linalg.norm(trans_real - trans_approx)
        # trans_arr[idx] = translation_error
        # angular_arr [idx] = angular_error
        # print("angular abs. error (mean degrees)", angular_error)
        # print("trans error", translation_error)
        # if (idx % 20 == 0 ):
        #     saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
        #     save_name = './check_rigid/source_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        #     sitk.WriteImage(saved, save_name)
            
        #     saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
        #     save_name = './check_rigid/target_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        #     sitk.WriteImage(saved, save_name)

        #     saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
        #     save_name = './check_rigid/rigid_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        #     sitk.WriteImage(saved, save_name)
   
    print ("training loss:", total)  

    for idx, image_data in enumerate(valloader):
        source, temp=image_data
        b = source.shape[0]
        source = source.to(dev).float() 
        # target =target.to(dev).float() 
      
        # print (source.shape)
        optimizer.zero_grad()   

        rx_val = random.randint(low_val, high_val)
        ry_val = random.randint(low_val, high_val)
        rz_val = random.randint(low_val, high_val)

        tx_val = random.randint(low_val, high_val)
        ty_val = random.randint(low_val, high_val)
        tz_val = random.randint(low_val, high_val)
        print ("epoch: ", epoch, "iter: ", idx, "rotation x:", rx_val, "rotation y:", ry_val, "rotation z:", rz_val, "translation x:", tx_val, "translation y:", ty_val, "translation z:", tz_val)
        mat = ci3d.create_transform(
            rx=rx_val, ry=ry_val, rz=rz_val,
            tx=2.0*tx_val/IMG_SIZE[0], ty=2.0*ty_val/IMG_SIZE[1], tz=2.0*tz_val/IMG_SIZE[2]
        )

        mat = mat[np.newaxis,:,:]
        mat = mat[:,0:3,:]
        mat = torch.tensor(mat).float()


        grids = torch.nn.functional.affine_grid(mat, [1,1] + IMG_SIZE).to(dev)
        target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=False)
        # # Start the timer
        start_time = time.time()
        xfm_1to2 = net_obj.forward((source,target))
        # Stop the timer
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        
        loss_val = loss_func(mat.to(dev), xfm_1to2 )

        predicted_grids = torch.nn.functional.affine_grid(xfm_1to2, [1,1] + IMG_SIZE)
        x_aligned = F.grid_sample(source,
                                  grid=predicted_grids,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)
     




        if (para.model.deformable == True):
            loss_val = NCC().loss(target, x_aligned)
            disp, deformed = diff_net(x_aligned, target, registration = True)
            Reg = Grad( penalty= 'l2')
            loss_reg = Reg.loss(disp)
            loss_dist_deform = NCC().loss(target, deformed)
            loss_val = loss_val + def_weight*loss_dist_deform + def_weight*loss_reg
            running_loss += loss_val.item()
            total_val += running_loss
        '''Print angular and tanslational error with or without deformable model '''
        '''To be implemented'''
        # print ("epoch: ", epoch, " batch loss: ", running_loss)
        running_loss = 0.0

        rot_real = mat[:,0:3,0:3].detach().cpu().numpy()
        trans_real = mat[:,0:3,3:].detach().cpu().numpy()
        
        rot_approx = xfm_1to2[:,0:3,0:3].detach().cpu().numpy()
        trans_approx = xfm_1to2[:,0:3,3:].detach().cpu().numpy()
        
        angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
        angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
        
        angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
        translation_error = np.linalg.norm(trans_real - trans_approx)
        trans_arr[idx] = translation_error
        angular_arr [idx] = angular_error
        print("angular abs. error (mean degrees)", np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx)))))
        print("trans error", np.linalg.norm(trans_real - trans_approx))
        
        if (idx % 20 == 0 ):
            saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
            save_name = './check_rigid_def_fetal/source_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
            sitk.WriteImage(saved, save_name)
            
            saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
            save_name = './check_rigid_def_fetal/target_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
            sitk.WriteImage(saved, save_name)

            saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
            save_name = './check_rigid_def_fetal/rigid_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
            sitk.WriteImage(saved, save_name)

            if (para.model.deformable == True):
                saved= sitk.GetImageFromArray(np.array(deformed[0,0,:,:,:].detach().cpu()))
                save_name = './check_rigid_def_fetal/deformed_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
                sitk.WriteImage(saved, save_name)

                velo = disp[0,:,:,:,:].reshape(3, 96, 96, 96).permute(1, 2, 3, 0)
                velo = velo.detach().cpu().numpy()
                save_path = './check_rigid_def_fetal/disp_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
                sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=True), save_path, False)
    np.save('trans_error_withoutdef.npy', trans_arr)
    np.save('angular_error_withoutdef.npy', angular_arr)
    print ("validation loss:", total_val)  