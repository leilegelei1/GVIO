import os
import pyproj
import numpy as np
from geo import *

data_root = '/home/jerry/Downloads/2011_10_03_drive_0027_extract/2011_10_03/2011_10_03_drive_0027_extract'

calib_I2V = os.path.join(data_root,'calib','calib_imu_to_velo.txt')
calib_V2C = os.path.join(data_root,'calib','calib_velo_to_cam.txt')

img_root = os.path.join(data_root,'image_00')
oxts_root = os.path.join(data_root,'oxts')

mat_I2V,mat_V2C = np.eye(4),np.eye(4)

lines_I2V = open(calib_I2V).readlines()
mat_I2V[:3,:3] = np.array(lines_I2V[1].strip('\n').split(' ')[1:]).astype(np.float).reshape([3,3])
mat_I2V[:3,3] =  np.array(lines_I2V[2].strip('\n').split(' ')[1:]).astype(np.float)
print(mat_I2V)

lines_V2C = open(calib_V2C).readlines()
mat_V2C[:3,:3] = np.array(lines_V2C[1].strip('\n').split(' ')[1:]).astype(np.float).reshape([3,3])
mat_V2C[:3,3] =  np.array(lines_V2C[2].strip('\n').split(' ')[1:]).astype(np.float)
print(mat_V2C)

mat_C2I = np.linalg.inv(mat_V2C.dot(mat_I2V))

R_C2I = mat_C2I[:3,:3]
print('--------------')
print(R_C2I.T.dot(R_C2I))
print(R_C2I.dot(np.array([1,0,0])))
print(mat_C2I.dot(np.array([1,0,0,1])))
print('--------------')

print(mat_C2I)
print(','.join(mat_C2I[:3,:3].reshape([-1]).astype(np.str).tolist()))
print(','.join(mat_C2I[:3,3].reshape([-1]).astype(np.str).tolist()))


def t2s(t):
    t,subt = t.split('.')
    h,m,s = t.strip().split(":")
    sec = int(h) * 3600 + int(m) * 60 + int(s)
    sec = str(sec) +'.' +  subt
    return round(float(sec),12)

base = '12:55:37.007950381'
base = t2s(base)
print(base)

from glob import glob
lines_img = open(os.path.join(img_root,'timestamps.txt')).readlines()
#img_names = os.listdir(os.path.join(img_root,'data'))
img_names = glob(os.path.join(img_root,'data','*.png'))
img_names.sort()
print(len(lines_img),len(img_names))

lines_oxts = open(os.path.join(oxts_root,'timestamps.txt')).readlines()
oxts_names = glob(os.path.join(oxts_root,'data','*.txt'))
oxts_names.sort()
print(len(lines_oxts),len(oxts_names))

output = os.path.join(data_root,'converted')
if not os.path.isdir(output):
    os.makedirs(output)

cam_start_line,oxts_start_line = 0,0
for ix,line in enumerate(lines_img):
    #print(line)
    if t2s(line.strip('\n').split(' ')[-1]) - base >= 0:
        cam_start_line = ix
        break
for ix,line in enumerate(lines_oxts):
    if t2s(line.strip('\n').split(' ')[-1]) - base >= 0:
        oxts_start_line = ix
        break
print(cam_start_line,oxts_start_line)

kitti_imgs = open(os.path.join(output,'kitti_imgs.txt'),'w')
kitti_imgs.write('Time img_path\n')
for ix in range(cam_start_line,len(img_names)):
    time = '%.9f' % t2s(lines_img[ix].strip('\n').split(' ')[-1])
    kitti_imgs.write('{} {}\n'.format(time,img_names[ix]))

kitti_imus = open(os.path.join(output,'kitti_imus.txt'),'w')
kitti_gps = open(os.path.join(output,'kitti_gps.txt'),'w')

kitti_imus.write('Time dt accelX accelY accelZ omegaX omegaY omegaZ\n')
kitti_gps.write('Time,X,Y,Z\n')

reference = None

for counter,ix in enumerate(range(oxts_start_line,len(lines_oxts)-1)):
    time = '%.9f' % t2s(lines_oxts[ix].strip('\n').split(' ')[-1])
    time_next = '%.9f' % t2s(lines_oxts[ix+1].strip('\n').split(' ')[-1])
    data = open(oxts_names[ix]).read()
    data = [float(i) for i in data.strip('\n').split(' ')]
    lat,lon,alt,ax,ay,az,wx,wy,wz = data[0],data[1],data[2],data[11],data[12],data[13],data[17],data[18],data[19]
    if reference is None:
        reference = [lat,lon,alt]

    st = '{} {} {} {} {} {} {} {}\n'.format(time,float(time_next)-float(time),ax,ay,az,wx,wy,wz)
    kitti_imus.write(st)

    if counter % 10 == 0:
        enu = geodetic_to_enu(lat,lon,alt,reference[0],reference[1],reference[2])
        #print(lat,lon,alt,reference,enu)
        st = '{} {} {} {}\n'.format(time,enu[0],enu[1],enu[2])
        kitti_gps.write(st)



