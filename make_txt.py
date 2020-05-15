import os, sys, pdb, numpy
from PIL import Image, ImageChops, ImageOps, ImageDraw
from os import listdir, getcwd

import shutil
#Real/subset/
wd=getcwd()
image_path=os.path.join(wd,'dataset','images')
#For train:
input_image1_path=os.path.join(image_path,'Train','T0')
input_image2_path=os.path.join(image_path,'Train','T1')
input_label_path=os.path.join(image_path,'Train','Ground_truth')
out_file=('./dataset/train.txt')

#For Val:
input_image1_path_val=os.path.join(image_path,'Val','T0')
input_image2_path_val=os.path.join(image_path,'Val','T1')
input_label_path_val=os.path.join(image_path,'Val','Ground_truth')
out_file_2=('./dataset/val.txt')

'''
#For Test:
input_image1_path_test=os.path.join(image_path,'Test','T0')
input_image2_path_test=os.path.join(image_path,'Test','T1')
input_label_path_test=os.path.join(image_path,'Test','Ground_truth')
out_file_3=('./test.txt')
'''

if __name__ == "__main__":
   #write train
  out_file = open(out_file, 'w')
  file_image1_name = [name for name in os.listdir(input_image1_path) if os.path.isfile(os.path.join(input_image1_path, name))]
  print(len(file_image1_name))
  for k in file_image1_name:
    A=os.path.join(input_image1_path,k)
    B=os.path.join(input_image2_path,k)
    Label=os.path.join(input_label_path,k)
    out_file.write(A+' '+B+' '+Label+' '+k+'\n')
  out_file.close()
  
  #write Val
  out_file_2 = open(out_file_2, 'w')
  file_image1_name = [name for name in os.listdir(input_image1_path_val) if os.path.isfile(os.path.join(input_image1_path_val, name))]
  print(len(file_image1_name))
  for k in file_image1_name:
    A_2=os.path.join(input_image1_path_val,k)
    B_2=os.path.join(input_image2_path_val,k)
    Label_2=os.path.join(input_label_path_val,k)
    out_file_2.write(A_2+' '+B_2+' '+Label_2+' '+k+'\n')
  out_file_2.close()
  ''' 
  #write Test
  out_file_3 = open(out_file_3, 'w')
  file_image1_name = [name for name in os.listdir(input_image1_path_test) if os.path.isfile(os.path.join(input_image1_path_test, name))]
  print(len(file_image1_name))
  for k in file_image1_name:
    A_3=os.path.join(input_image1_path_test,k)
    B_3=os.path.join(input_image2_path_test,k)
    Label_3=os.path.join(input_label_path_test,k)
    out_file_3.write(A_3+' '+B_3+' '+Label_3+' '+k+'\n')
  out_file_3.close()
  '''
