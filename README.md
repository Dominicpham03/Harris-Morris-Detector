

Moravec Detector

- I first created two array which corresponds to the eight point where the window would be shifted. 
I then created a nested for loop which iterate through the image. Where we calculate the anchor window against the shifted 
window to get the intesity value of the center matrix. I would then square each results and store them into a list where I 
find the minimum value and replace the center pixel with its value. I then created a threshold where if the minimum exceed 2600 
I would considder it as a keypoiunt and append it to a list which then I return the list.

LBP
- creating a kernel size of 15 x 15 where I would extract the LBP of each pixel within a 15 by 15 neighborhood. I would use a helper function which extracts the 
neighborhood of the pixel where we are creating a feature vector for. I would then iterate through the neighborhood and compute the LBP in a 3 x 3 window which I would
then append the value into a list. Aftering iterating through the 15 x 15 window I would then create a histogram that ranges between 0-255 which represent the intesity
value of pixel. I would then flatten the histogram and return the feature vector 
HOG

- For histogram of gradient I first applied the sobel filter to generate a x and y derv image which I would then create two for loop that iterate through the image.
Next I would compute the cthe graident orientation. I would then create a histogram which is composed of the graident oreitation which would then return me a feature 
vector of graident orientation of a pixel where the orientation spans from 0 - 180 where I would perform a ceiling operation and absolute to prevent any negative value 
and create whole value.


Feature Matching
- I created and if statement which determines wheter I was using HOG or LBP and Moravec or Harris detector. I would first compute the intrest point of both images
depending on whether the user decided to pick Moravec or Harris. I then would compute the HOG of the results of the Harris detector or Moravec which then I would
compare each feature vector to see if they match. For LBP I would compute the euclidean distance to see if there is a straight line distance between two points which 
can be used to quantify the similarity between two vector. I would then create a threshold which would check wheter what pixel is accepting our not to match with the other image.
If they do I would append it towards a list and then I would return the list where it is formated as x1 y1 x2 y2 which would 
help me to plot the matches later on.


Harris Detector 
- For the Harris Detector I first used cv2 to perform a gaussian blur on my image which I then applied the sobel filter onto my gaussian image.
I then created a nested for loop which iterated my for loop to created two dervative images which I then used to compute the Hessian Matrix which 
composed of Fx^2 fy2 and Fxy. I would then perform a gaussian blue on all derv images. I created another nested for loop which computed to find the 
value r which is the det-ktrace^2 which I would then use as a way to threshold my images for intrest points. In order to find my threshold value I perform 
trial and error to get the best possible results. Which I used a if statement to accept corner that exceed my threshold which I would then append it to a list 
and return the list of cordinates. 
