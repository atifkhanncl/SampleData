#use this
from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

def show(arr):
    im = Image.fromarray(np.array(255*arr/np.max(arr),dtype=np.uint8))
    im.show()

def getcentre(contours):
    M = cv2.moments(contours)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return((cX,cY))

def drawcentres(img_msk_blend, arr,contours,labels=[],ptrad=3):
    '''Modified function from mitocyto to write number and centres on contours'''
    rgb = img_msk_blend
    h,w = arr.shape
    uselabs = True 
    
    for i,cnt in enumerate(contours):
        cX,cY = getcentre(cnt)
        cv2.circle(rgb, (cX,cY), ptrad, (255, 0, 0), -1)
        if uselabs and arr.shape[1]>100:
            cv2.putText(rgb, str(i), (min(w-10,max(10,cX - 20)), min(h-10,max(10,cY - 10))),cv2.FONT_HERSHEY_SIMPLEX, 0.5*arr.shape[1]/2656.0, (255, 255, 255), 1)
          
    return(rgb)

def get_over_the_border_area(img, annarr, element_size = 2):   # set element size to 2 for IMC and 6 for QIF images
    '''Function that returns results: list with all our metrics and fibre number, and contours'''
    
    annarr[0,:-1] = annarr[:-1,-1] = annarr[-1,::-1] = annarr[-2:0:-1,0] = annarr.max()

    # Find fibres from annotation
    contours,hierarchy = cv2.findContours(annarr, cv2.RETR_CCOMP,2)
    # Ensure only looking at holes inside contours...
    contours = [c for i,c in enumerate(contours) if hierarchy[0][i][3] != -1]
                                          
    # Element for erosion/dilation
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * element_size + 1, 2 * element_size + 1),
                                        (element_size, element_size))

       
    results = [] 
    for i,contour in enumerate(tqdm((contours), desc='Contours processed')):
        black = np.zeros(annarr.shape,dtype=np.uint8)
        cv2.drawContours(black,[contour], -1, (255), -1)
        edge_eroded = black - cv2.erode(black,element) # area to check for dystrophin +ve
        edge_dilated = cv2.dilate(black,element) - black  # area to check for VDAC +ve
        
        # calculating pixels in three area 1) dilated area 2) eroded area and 3) fibre area - eroded area
        outer_pixels = edge_dilated == 255
        inner_pixels = edge_eroded == 255
        fibre_pixels = black == 255
        check_area = outer_pixels + inner_pixels
        #show(check_area)
        #show(inner_pixels)

        # THERSHOLD .85 & .75 for QIF
        membrane_thresh = img[:,:,2] > np.quantile(img[:,:,2],0.85)
        mito_mass_thresh = img[:,:,1] > np.quantile(img[:,:,1],0.75)
        #show(mito_mass_thresh)
        #show(membrane_thresh)
        
        # calculating red(dystophin) and green(VDAC) pixels in appropriate areas
        membrane_included = np.logical_and(membrane_thresh, inner_pixels)
        mito_mass_missed = np.logical_and(mito_mass_thresh, outer_pixels)
        membrane_in_fibre = np.logical_and(membrane_thresh, fibre_pixels)
        
        # calculating proportions
        proportion_membrane_included = (np.sum(membrane_included)/np.sum(outer_pixels))*100
        proportion_mito_mass_missed = (np.sum(mito_mass_missed)/np.sum(inner_pixels))*100
        proportion_membrane_in_fibre = (np.sum(membrane_in_fibre)/np.sum(fibre_pixels))*100
        
        values = [i,proportion_membrane_included,proportion_mito_mass_missed,proportion_membrane_in_fibre]
        results.append(values)
        
        
    
    return(results,contours)

def are_same_fibres(cont1, cont2):
    c1X,c1Y = getcentre(cont1)
    #c2X,c2Y = getcentre(cont2)
    dist = cv2.pointPolygonTest(cont2,(c1X,c1Y),True) # checking if centre of first contour exist in second contour
    if dist > 0:
        val = True
    else:
        val = False
    return val

def IoU_two_contours(ann,cont1, cont2):
    contours = [cont1, cont2]

        # Create image filled with zeros the same size of original image
    blank = np.zeros(ann.shape[0:2])

        # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), contours, 0, 255,thickness = cv2.FILLED) # 0 index contour i.e. first one
    image2 = cv2.drawContours(blank.copy(), contours, 1, 255,thickness = cv2.FILLED) # 1 index contour i.e. second one

        # Use the logical AND operation on the two images
        # Since the two images had bitwise and applied to it,
        # there should be a '1' or 'True' where there was intersection
        # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    IoU = np.sum(intersection)/np.sum(union)    
    img_msk_blend = cv2.merge((blank, image1, image2))      
    return IoU, img_msk_blend

def IoU_score (annarr, mask1, mask2):
    
    annarr[0,:-1] = annarr[:-1,-1] = annarr[-1,::-1] = annarr[-2:0:-1,0] = annarr.max()
    mask1[0,:-1] = mask1[:-1,-1] = mask1[-1,::-1] = mask1[-2:0:-1,0] = mask1.max()
    mask2[0,:-1] = mask2[:-1,-1] = mask2[-1,::-1] = mask2[-2:0:-1,0] = mask2.max()
    # Find fibres from mask1
    contours1,hierarchy1 = cv2.findContours(mask1, cv2.RETR_CCOMP,2)
    # Ensure only looking at holes inside contours...
    contours1 = [c for i,c in enumerate(contours1) if hierarchy1[0][i][3] != -1]
    
    # Find fibres from mask2
    contours2,hierarchy2 = cv2.findContours(mask2, cv2.RETR_CCOMP,2)
    # Ensure only looking at holes inside contours...
    contours2 = [c for i,c in enumerate(contours2) if hierarchy2[0][i][3] != -1]                                      
    
    
    
    blend_images =[] # stores images of contours overlapping
    results = [] 
    for i,contour in enumerate(tqdm((contours1), desc='Contours processed')):
        for j, cont in enumerate(contours2):
            
            val = are_same_fibres(contours1[i],contours2[j])
            if val:
                IoU, img_blend = IoU_two_contours(annarr,contours1[i], contours2[j])
                values = [i,j,IoU]
                results.append(values)
                blend_images.append(img_blend)
    return results, blend_images
                
                
            
    

inp_img_dir = './images'   # directory with orginal images to be annotated
inp_msk_dir = './masks'   # directory with annotation mask done by Apeer
inp_our_msk_dir = './ourMasks'# directory with annotation mask done by us
out_dir = './outputs'# output directory


img_list = [f for f in
            os.listdir(inp_img_dir)
            if os.path.isfile(os.path.join(inp_img_dir, f))]

for infile in img_list:
    infile_img_path = os.path.join(inp_img_dir, infile)
    infile_msk_path = os.path.join(inp_msk_dir, infile.split('.')[0] + '.tif')
    infile_msk_our_path = os.path.join(inp_our_msk_dir, infile.split('.')[0] + '.png')
    infile_out_path = os.path.join(out_dir, infile.split('.')[0] + '.csv')
    infile_out_path_IoU = os.path.join(out_dir, infile.split('.')[0] + '_IoU.csv')
    infile_out_path_img = os.path.join(out_dir, infile.split('.')[0] + '.tif')
   
    img = cv2.imread(infile_img_path)
    mask = cv2.imread(infile_msk_path,0)
    mask_our = cv2.imread(infile_msk_our_path,0)
    
    annarr = np.array(mask,dtype=np.uint8)
    thresh,annarr = cv2.threshold(annarr, 0, 255, cv2.THRESH_BINARY)
    # inverting binary mask so as to have white pixels as annotations
    annarr_invert = (255-annarr)

    # Adding mask as blue channel. Fine.  But what about when have nuclear signal in blue channel?
    img_msk_blend = cv2.merge((annarr_invert, cv2.split(img)[1], cv2.split(img)[2]))
    
    # getting membrane included and mass missed metrics
    results_metrics,contours = get_over_the_border_area(img, mask)
    
    # getting IoU between our and Apeer annotations
    results_IoU, blend_images = IoU_score (annarr, mask, mask_our)
   
    # identifying the contours
    result_img=drawcentres(img_msk_blend, mask, contours)
    cv2.imwrite(infile_out_path_img, result_img)
    
    # saving images of all IOU contours
    for i, img in enumerate(blend_images):
        path = os.path.join(out_dir, infile.split('.')[0] + '_'+ str(i)+'_.png')
        cv2.imwrite(path, img)
    
   
    header = ['FibreNumber','proportion_membrane_included', 'proportion_mito_mass_missed', 'proportion_membrane']
   
    with open(infile_out_path, "w", encoding="UTF8",newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(results_metrics)
    
    header1 = ['FibreApeer','FibreUs', 'IoU Score']
    with open(infile_out_path_IoU, "w", encoding="UTF8",newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header1)
        # write the data
        writer.writerows(results_IoU)