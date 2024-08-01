

import cv2
import numpy as np
import math
import queue
from PIL import Image



def load_img(file_name):
    try: 
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        
        return image
    except Exception as e:
        print("Error loading image:", e)
        return None

def display_img(image):
    if image is not None:
        windowName = 'image'
        cv2.imshow(windowName, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image")


def get_neighborhood(image, x, y):
    neighborhood = np.zeros((3,3))
    for i in range(max(0, x - 1), min(len(image), x + 2)):
        for j in range(max(0, y - 1), min(len(image[0]), y + 2)):
            neighborhood[i-(x-1)][j-(y-1)] = image[i][j]
    return neighborhood

def harris_detector(image):
    
    padded_image = np.pad(image, 1, 'constant', constant_values=0)
    image = cv2.GaussianBlur(image, (3,3), 1)
    gradient_x = np.zeros_like(image)
    gradient_y = np.zeros_like(image)
    cord = []
    

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = sobel_x.T

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+3, j:j+3]
            gradient_x[i][j] = np.sum(neighborhood * sobel_x)
            gradient_y[i][j] = np.sum(neighborhood * sobel_y)
    
    fx_2 = gradient_x**2
    fy_2 = gradient_y**2
    fxy = gradient_x*gradient_y
    fx_2 = cv2.GaussianBlur(fx_2, (3,3), 1)
 
    fy_2 = cv2.GaussianBlur(fy_2, (3,3), 1)
    fxy  = cv2.GaussianBlur(fxy , (3,3), 1)
    k = .06
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            H = np.array([[fx_2[i, j], fxy[i, j]], [fxy[i, j], fy_2[i, j]]])


            
            det = np.linalg.det(H)
            trace = np.trace(H)

            
            corner= det - k * (trace ** 2)
            
            if(corner > 7000):
                 print(corner)
                 cord.append([i,j])
                   
    return cord


    

def moravec_detector(image):
   
    keypoint = []
    x_cord = [-1,-1,0,-1,1,0,1,1]
    y_cord = [-1,0,-1,1,0,1,-1,1]
    total = []
    width = image.shape[0]
    height = image.shape[1]
    image = np.pad(image, 2, 'constant', constant_values=0)
    
    


    for i in range(2,width):
        for j in range(2,height):

            for k in range(7):
                center =  get_neighborhood(image,i,j)
                window = get_neighborhood(image,i - x_cord[k],j-y_cord[k])
                x =   np.subtract(window, center)
                square_matrix = np.dot(x,x)
                total.append(np.sum(square_matrix ))
            min = np.min(total)
            total.clear()
            if min > 2400:
                keypoint.append([i,j])

    return keypoint



def plot_keypoints(image, keypoints):
    image_with_keypoint = np.copy(image)
    image_with_keypoint = cv2.cvtColor(image_with_keypoint, cv2.COLOR_GRAY2RGB)
     
    for point in keypoints:
        x, y = point
        image_with_keypoint[int(x), int(y)] = [0, 0, 255]
        
        
    return image_with_keypoint  

def findLBP(neighborhood):
    dec_value = 0
    if(neighborhood[1,1] < neighborhood[0,0]):
            dec_value += 128
    if(neighborhood[1,1] < neighborhood[0,1]):
            dec_value += 64
    if(neighborhood[1,1] < neighborhood[0,2]):
            dec_value += 32
    if(neighborhood[1,1] < neighborhood[1,2]):
            dec_value += 16
    if(neighborhood[1,1] < neighborhood[2,2]):
            dec_value += 8
    if(neighborhood[1,1] < neighborhood[2,1]):
            dec_value += 4
    if(neighborhood[1,1] < neighborhood[2,0]):
            dec_value += 2
    if(neighborhood[1,1] < neighborhood[1,0]):
            dec_value += 1
    return dec_value
 


def extract_LBP(image, keypoint):
    x, y = keypoint
    size = 15  
    neighborhood = np.empty((size, size))
    value = []
    
    start_x = max(0, x - size // 2)
    end_x = min(image.shape[0], x + size // 2 + 1)
    start_y = max(0, y - size // 2)
    end_y = min(image.shape[1], y + size // 2 + 1)
    
    # Extract the neighborhood
    neighborhood = image[start_x:end_x, start_y:end_y]
    neighborhood = np.pad(neighborhood, 2, 'constant', constant_values=0)
    for i in range(2, neighborhood.shape[0] - 2):
        for j in range(2, neighborhood.shape[1] - 2):
            
            sub_neighborhood = neighborhood[i - 1:i + 2, j - 1:j + 2]
        
            value.append(findLBP(sub_neighborhood))
    
    
    hist, bins = np.histogram(value, bins=range(0, 256))
    feature_vector = hist.flatten()
    return feature_vector





def apply_filter(filter,image):
  applied = np.zeros((3,3))
  applied = filter * image
  return applied


def extract_HOG(image,keypoint):
    x,y = keypoint
    graident = []
    orientation = []
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    neighborhood = get_neighborhood(image,x,y)
    
  
    for k in range(3):
        for l in range(3):
                derv_x = apply_filter(sobel_x,neighborhood)
                derv_y = apply_filter(sobel_y,neighborhood)
                graident.append(np.hypot(derv_x[k][l],derv_y[k][l]))
                orientation.append(np.absolute(np.ceil(np.degrees(np.arctan2(derv_x[k][l],derv_y[k][l])))))
    hist,bin_edges = np.histogram(orientation,bins=range(0,181))
    feature_vector = hist.flatten()
    
         



    return feature_vector

def feature_matching(image1, image2, detector, extractor):
    sim_cord = []

    if detector == "Moravec":
        image1_keypoints = moravec_detector(image1)
        image2_keypoints = moravec_detector(image2)
    elif detector == "Harris":
        image1_keypoints = harris_detector(image1)
        image2_keypoints = harris_detector(image2)
    else:
        print("Not an Option")
        return 0

    if extractor == "LBP":
     descriptors1 = [extract_LBP(image1, kp) for kp in image1_keypoints]
     descriptors2 = [extract_LBP(image2, kp) for kp in image2_keypoints]

     for i, desc1 in enumerate(descriptors1):
          for j, desc2 in enumerate(descriptors2):
           
            euclidean_distance = np.linalg.norm(desc1-desc2)
            print(euclidean_distance)
            if 20 < euclidean_distance < 23:
                 x1, y1 = image1_keypoints[i]
                 x2, y2 = image2_keypoints[j]
                 sim_cord.append([x1, y1, x2, y2])



 
    elif extractor == "HOG":
        descriptors1 = [extract_HOG(image1, kp) for kp in image1_keypoints]
        descriptors2 = [extract_HOG(image2, kp) for kp in image2_keypoints]
        for i, desc1 in enumerate(descriptors1):
         for j, desc2 in enumerate(descriptors2):
           
            if np.array_equal(desc1, desc2):
                x1, y1 = image1_keypoints[i]
                x2, y2 = image2_keypoints[j]
                sim_cord.append([x1, y1, x2, y2])


    else:
        print("Not an Option")
        return 0



    return sim_cord





def plot_matches(image1, image2, matches):
    max_height = max(image1.shape[0], image2.shape[0])
    
    padded_image1 = np.pad(image1, ((0, max_height - image1.shape[0]), (0, 0)), mode='constant')
    padded_image2 = np.pad(image2, ((0, max_height - image2.shape[0]), (0, 0)), mode='constant')
    
    image_with_keypoint_one = cv2.cvtColor(padded_image1, cv2.COLOR_GRAY2RGB)
    image_with_keypoint_two = cv2.cvtColor(padded_image2, cv2.COLOR_GRAY2RGB)
    
    combined_image = np.hstack((image_with_keypoint_one, image_with_keypoint_two))
    
    for point in matches:
        x1, y1, x2, y2 = point
        
        combined_image[int(x1), int(y1), :] = [0, 0, 255]
        combined_image[int(x2), int(y2) + image_with_keypoint_one.shape[1], :] = [0, 0, 255]
        
        cv2.line(combined_image, (int(y1), int(x1)), (int(y2) + image_with_keypoint_one.shape[1], int(x2)), (0, 0, 255), 1, lineType=cv2.LINE_AA)
        
        
        cv2.circle(combined_image, (int(y1), int(x1)), 3, (0, 0, 255), -1)
        cv2.circle(combined_image, (int(y2) + image_with_keypoint_one.shape[1], int(x2)), 3, (0, 0, 255), -1)
    
    return combined_image




filePath = "cat.jpg"
filePath2 = "cat.jpg"
image = load_img(filePath)
image2 = load_img(filePath2)


# cord = harris_detector(image)
# image_plot = plot_keypoints(image,cord)
# display_img(image_plot)

# cord = moravec_detector(image)
# image_plot = plot_keypoints(image,cord)
# display_img(image_plot)





# # print("LBP")
# matches = feature_matching(image,image2,"Moravec","LBP")
# image_compare , image_compare2= plot_matches(image,image2,matches)  
# display_img(image_compare)
# display_img(image_compare2)

# print("HOG")
# matches = feature_matching(image,image2,"Moravec","HOG")
# image_compare , image_compare2= plot_matches(image,image2,matches)  
# display_img(image_compare)
# display_img(image_compare2)



# # print("LBP")
# matches = feature_matching(image,image2,"Moravec","LBP")
# image_compare , image_compare2= plot_matches(image,image2,matches)  
# display_img(image_compare)
# display_img(image_compare2)

# # print("HOG")
# matches = feature_matching(image,image2,"Moravec","HOG")
# image_compare , image_compare2= plot_matches(image,image2,matches)  
# display_img(image_compare)
# display_img(image_compare2)












