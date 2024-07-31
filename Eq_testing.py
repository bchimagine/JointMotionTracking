import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random 
import numpy as np

import custom_image3d as ci3d
from SitkDataSet import SitkDataset as SData
from model import ConvNetFC
import rxfm_net
import registration_tools as rt
import SimpleITK as sitk
import subprocess
from pytorch3d import transforms as pt3d_xfms
# Load the saved model checkpoint
checkpoint = torch.load('our_method.pth')
checkpoint_deform = torch.load('our_method_2.pth')
checkpoint_km = torch.load('keymorph.pth')
checkpoint_ours = torch.load('./best_model_29.pth')
# Set device (GPU or CPU)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)
h_dims = [4, 8, 8, 16, 16, 32, 32, 64]
dim = 3
key_num = 128
low = 1
high = 2
IMG_SIZE = [96,96,96]
# Create the networks and load their states
net_rigid_s = ConvNetFC(dim, 1, h_dims, key_num * dim, norm_type='batch')
net_rigid_t = ConvNetFC(dim, 1, h_dims, key_num * dim, norm_type='batch')
net_rigid_s.load_state_dict(checkpoint['net_rigid_s_state_dict'])
net_rigid_t.load_state_dict(checkpoint['net_rigid_t_state_dict'])
# Move the models to the device (e.g., 'cuda')
net_rigid_s.to(dev)
net_rigid_t.to(dev)




net_rigid_s_df = ConvNetFC(dim, 1, h_dims, key_num * dim, norm_type='batch')
net_rigid_t_df = ConvNetFC(dim, 1, h_dims, key_num * dim, norm_type='batch')
net_rigid_s_df.load_state_dict(checkpoint_deform['net_rigid_s_state_dict'])
net_rigid_t_df.load_state_dict(checkpoint_deform['net_rigid_t_state_dict'])
# Move the models to the device (e.g., 'cuda')
net_rigid_s_df.to(dev)
net_rigid_t_df.to(dev)




net_rigid_s_km = ConvNetFC(dim, 1, h_dims, key_num * dim, norm_type='batch')
net_rigid_t_km = ConvNetFC(dim, 1, h_dims, key_num * dim, norm_type='batch')
net_rigid_s_km.load_state_dict(checkpoint_km['net_rigid_s_state_dict'])
net_rigid_t_km.load_state_dict(checkpoint_km['net_rigid_t_state_dict'])
# Move the models to the device (e.g., 'cuda')
net_rigid_s_km.to(dev)
net_rigid_t_km.to(dev)


n_conv_chan = 1
n_chan = 64
overfit = True 
net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)
net_obj.load_state_dict(checkpoint_ours['eq_tracking_state_dict'])
net_obj = net_obj.to(dev)



# Now, you can use model_rigid_s and model_rigid_t for testing.
json_file_val = './data_squ_test.json'
dataset_val = SData(json_file_val, "test")
valloader = DataLoader(dataset_val, batch_size= 1, shuffle=False)

trans_arr_our = np.zeros(len(valloader))
angular_arr_our=  np.zeros(len(valloader))

trans_arr_our2 = np.zeros(len(valloader))
angular_arr_our2 =  np.zeros(len(valloader))

trans_arr_km = np.zeros(len(valloader))
angular_arr_km =  np.zeros(len(valloader))

trans_arr_cv = np.zeros(len(valloader))
angular_arr_cv =  np.zeros(len(valloader))

trans_arr_eq = np.zeros(len(valloader))
angular_arr_eq  =  np.zeros(len(valloader))


for idx, image_data in enumerate(valloader):
    source, target_real =image_data
    #Target_real is for real data
    b = source.shape[0]
    source = source.to(dev).float() 
    
    #Creating batch-wise online rigid transformation
    rx_train = random.randint(low, high)
    ry_train = random.randint(low, high)
    rz_train = random.randint(low, high)

    tx_train = random.randint(low, high)
    ty_train = random.randint(low, high)
    tz_train = random.randint(low, high)

    print ("rotation x:", rx_train, "rotation y:", ry_train, "rotation z:", rz_train, "translation x:", tx_train, "translation y:", ty_train, "translation z:", tz_train)
    
    #Creating batch-wise rigid matrix
    mtrx_batch = torch.zeros(b, 3, 4)
    mat = ci3d.create_transform(
        rx=rx_train, ry=ry_train, rz=rz_train,
        tx=2.0*tx_train/IMG_SIZE[0], ty=2.0*ty_train/IMG_SIZE[1], tz=2.0*tz_train/IMG_SIZE[2]
    )
    mat = mat[np.newaxis,:,:]
    mat = mat[:,0:3,:]
    mat = torch.tensor(mat).float()
    mtrx_batch [:] = mat 

    #Creating batch-wise simulated target 
    grids = F.affine_grid(mtrx_batch, source.size(), align_corners=False).to(dev)
    target = F.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=True)

    #Feed source and target into networks
    moving_kp, moving_latent= net_rigid_s(source)
    target_kp, target_latent= net_rigid_t(target)

    #Calculating the keypoints
    moving_kp = moving_kp.reshape(b, dim, key_num)
    target_kp = target_kp.reshape(b, dim, key_num)

    # Close form solution for Rigid 
    rigid_matrix = rt.close_form_rigid(moving_kp, target_kp)
    ''' TO BE implemented here: Affine to Rigid'''
    inv_matrix = torch.zeros(source.size(0), 4, 4)
    inv_matrix[:, :3, :4] = rigid_matrix  # affine_matrix
    inv_matrix[:, 3, 3] = 1
    inv_matrix = torch.inverse(inv_matrix)[:, :3, :]
    grid = F.affine_grid(inv_matrix,
                         source.size(),
                         align_corners=False)
    grid = grid.to(dev)

    #Align the image by the inverse rigid matrix 
    x_aligned = F.grid_sample(source,
                              grid=grid,
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=False)

    saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
    save_name = './result/ours1/source_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    
    saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
    save_name = './result/ours1/target_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
    save_name = './result/ours1/rigid_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)


    rot_real = mat[:,0:3,0:3].detach().cpu().numpy()
    trans_real = mat[:,0:3,3:].detach().cpu().numpy()
    
    rot_approx = rigid_matrix[:,0:3,0:3].detach().cpu().numpy()
    trans_approx = rigid_matrix[:,0:3,3:].detach().cpu().numpy()
    
    angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
    angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
    
    angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
    translation_error = np.linalg.norm(trans_real - trans_approx)
    trans_arr_our[idx] = translation_error
    angular_arr_our [idx] = angular_error


    moving_kp, moving_latent= net_rigid_s_df(source)
    target_kp, target_latent= net_rigid_t_df(target)

    #Calculating the keypoints
    moving_kp = moving_kp.reshape(b, dim, key_num)
    target_kp = target_kp.reshape(b, dim, key_num)

    # Close form solution for Rigid 
    rigid_matrix = rt.close_form_rigid(moving_kp, target_kp)
    ''' TO BE implemented here: Affine to Rigid'''
    inv_matrix = torch.zeros(source.size(0), 4, 4)
    inv_matrix[:, :3, :4] = rigid_matrix  # affine_matrix
    inv_matrix[:, 3, 3] = 1
    inv_matrix = torch.inverse(inv_matrix)[:, :3, :]
    grid = F.affine_grid(inv_matrix,
                         source.size(),
                         align_corners=False)
    grid = grid.to(dev)

    #Align the image by the inverse rigid matrix 
    x_aligned = F.grid_sample(source,
                              grid=grid,
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=False)

    saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
    save_name = './result/ours2/source_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    
    saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
    save_name = './result/ours2/target_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
    save_name = './result/ours2/rigid_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    rot_approx = rigid_matrix[:,0:3,0:3].detach().cpu().numpy()
    trans_approx = rigid_matrix[:,0:3,3:].detach().cpu().numpy()
    
    angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
    angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
    
    angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
    translation_error = np.linalg.norm(trans_real - trans_approx)
    trans_arr_our2[idx] = translation_error
    angular_arr_our2 [idx] = angular_error






    moving_kp, moving_latent= net_rigid_s_km(source)
    target_kp, target_latent= net_rigid_t_km(target)

    #Calculating the keypoints
    moving_kp = moving_kp.reshape(b, dim, key_num)
    target_kp = target_kp.reshape(b, dim, key_num)

    # Close form solution for Rigid 
    rigid_matrix = rt.close_form_rigid(moving_kp, target_kp)
    ''' TO BE implemented here: Affine to Rigid'''
    inv_matrix = torch.zeros(source.size(0), 4, 4)
    inv_matrix[:, :3, :4] = rigid_matrix  # affine_matrix
    inv_matrix[:, 3, 3] = 1
    inv_matrix = torch.inverse(inv_matrix)[:, :3, :]
    grid = F.affine_grid(inv_matrix,
                         source.size(),
                         align_corners=False)
    grid = grid.to(dev)

    #Align the image by the inverse rigid matrix 
    x_aligned = F.grid_sample(source,
                              grid=grid,
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=False)

    saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
    save_name = './result/keymorph/source_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    
    saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
    save_name = './result/keymorph/target_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
    save_name = './result/keymorph/rigid_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    rot_approx = rigid_matrix[:,0:3,0:3].detach().cpu().numpy()
    trans_approx = rigid_matrix[:,0:3,3:].detach().cpu().numpy()
    
    angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
    angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
    
    angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
    translation_error = np.linalg.norm(trans_real - trans_approx)
    trans_arr_km[idx] = translation_error
    angular_arr_km[idx] = angular_error





    saved = np.array(source[0,0,:,:,:].detach().cpu())
    normalized_array = ((saved - saved.min()) / (saved.max() - saved.min()) * 255).astype(np.uint8)
    saved_img= sitk.GetImageFromArray(normalized_array)
    save_src= './result/conv/source_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved_img, save_src)
    
    saved = np.array(target[0,0,:,:,:].detach().cpu())
    normalized_array = ((saved - saved.min()) / (saved.max() - saved.min()) * 255).astype(np.uint8)
    saved_img= sitk.GetImageFromArray(normalized_array)
    save_tar = './result/conv/target_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved_img, save_tar)

    save_align = './result/conv/rigid_' +  str(idx) + '.nii.gz'

    """ Replace this with the ImageRegistration8 from your local ITK repo! """
    exepath = '/home/jianwang/Packages/InsightToolkit-5.3.0/Examples/Wrapping/Generators/Python/itk/ImageRegistration8'
    subprocess.run([exepath, save_tar, save_src, save_align], check=True)
    rot_approx = rigid_matrix[:,0:3,0:3].detach().cpu().numpy()
    trans_approx = rigid_matrix[:,0:3,3:].detach().cpu().numpy()
    
    angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
    angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
    
    angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
    translation_error = np.linalg.norm(trans_real - trans_approx)
    trans_arr_cv[idx] = translation_error
    angular_arr_cv[idx] = angular_error

    xfm_1to2 = net_obj.forward((source,target))






    predicted_grids = torch.nn.functional.affine_grid(xfm_1to2, [1,1] + IMG_SIZE)
    x_aligned = F.grid_sample(source,
                              grid=predicted_grids,
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=False)
    saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
    save_name = './result/eq/source_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    
    saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
    save_name = './result/eq/target_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
    save_name = './result/eq/rigid_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    rot_real = mat[:,0:3,0:3].detach().cpu().numpy()
    trans_real = mat[:,0:3,3:].detach().cpu().numpy()
    
    rot_approx = xfm_1to2[:,0:3,0:3].detach().cpu().numpy()
    trans_approx = xfm_1to2[:,0:3,3:].detach().cpu().numpy()
    
    angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
    angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
    
    angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
    translation_error = np.linalg.norm(trans_real - trans_approx)
    trans_arr_eq[idx] = translation_error
    angular_arr_eq[idx] = angular_error



np.save("trans_our.npy", trans_arr_our)
np.save("angular_our.npy", angular_arr_our)


np.save("trans_our2.npy", trans_arr_our2)
np.save("angular_our2.npy", angular_arr_our2)
  

np.save("trans_cv.npy",  trans_arr_cv)
np.save("angular_cv.npy", angular_arr_cv)
 

np.save("trans_km.npy",  trans_arr_km)
np.save("angular_km.npy", angular_arr_km)

np.save("trans_eq.npy",  trans_arr_eq)
np.save("angular_eq.npy", angular_arr_eq)
