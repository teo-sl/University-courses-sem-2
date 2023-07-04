import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import torch.nn as nn
import math

"""
To enhance dimensionality => ndimage.zoom(img,X)
where, if X>1 you increase resolution using interpolation
if x<0 the resolution is reduced

for interpolation, order = 0 linear, order = 1 bilinear, order = 2 tricubic
plt.imshow(ndimage.zoom(img[200:250,190:240,0],2,order=0),cmap="gray")

Per identificare oggetti considera il thresholding


Applying a Gaussian blur to an image means doing a convolution of the Gaussian with the image. Convolution is associative: Applying two Gaussian blurs to an image is equivalent to convolving the two Gaussians with each other, and convolving the resulting function with the image.
https://computergraphics.stackexchange.com/questions/256/is-doing-multiple-gaussian-blurs-the-same-as-doing-one-larger-blur#:~:text=Applying%20a%20Gaussian%20blur%20to,resulting%20function%20with%20the%20image.
https://math.stackexchange.com/questions/3159846/what-is-the-resulting-sigma-after-applying-successive-gaussian-blur


batch norm e local response normalization:
- LRN is a non-trainable layer
In DNNs, the purpose of this lateral inhibition is to carry out local contrast enhancement so that locally maximum pixel values are used as excitation for the next layers.


exponential transformation: img_exp = 4*(((1+i)**img)-1)
"""

# corrisponde a fx
sobelH = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])
#corrisponde a fy
sobelV = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])
# laplaciano è derivata seconda, si applica gauss per fare smoothing
laplacian = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])
laplacian_2 = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])


def convolution(img,filter,p=0):
    """
    p is for padding
    """
    padded = np.zeros((img.shape[0]+2*p,img.shape[1]+2*p))
    padded[p:img.shape[0]+p,p:img.shape[1]+p] = img
    img = padded
    ret = np.zeros((img.shape[0],img.shape[1]))

    q1 = filter.shape[0]%2
    q2 = filter.shape[1]%2
    for i in range(filter.shape[0]//2,img.shape[0]-filter.shape[0]//2):
        for j in range(filter.shape[1]//2,img.shape[1]-filter.shape[1]//2):
            ret[i,j] = np.sum(img[i-filter.shape[0]//2:i+filter.shape[0]//2+q1,
                                  j-filter.shape[1]//2:j+filter.shape[1]//2+q2] * filter)
    return ret

def convolution_function(img,function,filter_size,p=0):
    """
    p is for padding

    Example of use:
    out_max = convolution_function(img,lambda x : np.max(x),kernel_size,p=1)
    """

    padded = np.zeros((img.shape[0]+2*p,img.shape[1]+2*p))
    padded[p:img.shape[0]+p,p:img.shape[1]+p] = img
    img = padded
    ret = np.zeros((img.shape[0],img.shape[1]))

    q1 = filter_size[0]%2
    q2 = filter_size[1]%2
    for i in range(filter_size[0]//2,img.shape[0]-filter_size[0]//2):
        for j in range(filter_size[1]//2,img.shape[1]-filter_size[1]//2):
            ret[i,j] = function(img[i-filter_size[0]//2:i+filter_size[0]//2+q1,
                                  j-filter_size[1]//2:j+filter_size[1]//2+q2])
    return ret

def read_gray_img(path):
    img = cv2.imread(path)
    if len(img.shape)==3:
        if img.shape[2]==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            img = img.reshape(img.shape[0],img.shape[1])
    return img

def read_image_rbg(path):
    img = cv2.imread(path)
    if len(img.shape)!=3 or img.shape[2]!=3:
        raise Exception('Immagine non a colori')
    return img[:,:,::-1]

def apply_affine(img,A,recenter=True):
    m,n = img.shape
    center  = np.array([m//2,n//2])
    new_center = A @ [*center,1]
    offset = (new_center[0:2]-center).astype(int)
    ret = np.zeros_like(img)

    for i in range(m):
        for j in range(n):
            new_pos = (A @ np.array([i,j,1])).astype(int)
            if recenter:
                new_pos = new_pos[0:2] - offset
            else:
                new_pos = new_pos[0:2]
            if new_pos[0]<0 or new_pos[0]>=m or new_pos[1]<0 or new_pos[1]>=n:
                continue
            ret[new_pos[0],new_pos[1]] = img[i,j]
    return ret

def apply_affine_complete(img,theta=0,tx=0,ty=0,cx=1,cy=1,sv=0,sh=0):
    A = np.eye(3)
    A[0,2] = tx
    A[1,2] = ty
    B = np.eye(3)
    B[0,0] = cx
    B[1,1] = cy
    theta = np.deg2rad(theta)
    C = np.array([
        [np.cos(theta),-np.sin(theta),0],
        [np.sin(theta),np.cos(theta),0],
        [0,0,1]
    ]) 
    D = np.eye(3)
    D[0,1] = sv
    E = np.eye(3)
    E[1,0] = sh
    mat = A @ B @ C @ D @ E
    ret = np.zeros_like(img)
    m,n = img.shape
    for i in range(m):
        for j in range(n):
            [x_,y_,_] = mat @ np.array([i,j,1])
            if x_>=0 and x_<m and y_>=0 and y_<n:
                ret[int(x_),int(y_)] = img[i,j]
    return ret


def draw_bbox(img,A,color=125,width=1):
    """
    A is the bbox coordinates in the standard form [x_min,y_min,x_max,y_max]
    """
    img = img.copy()
    img[A[0]:A[2],A[1]:A[1]+width] = color
    img[A[0]:A[2],A[3]:A[3]+width] = color

    img[A[0]:A[0]+width,A[1]:A[3]] = color
    img[A[2]:A[2]+width,A[1]:A[3]] = color
    return img


def compute_iou(A,B):
    """
    A and B are bbox coordinates in the standard form [x_min,y_min,x_max,y_max]
    """
    
    x1 = max(A[0],B[0])
    x2 = min(A[2],B[2])
    y1 = max(A[1],B[1])
    y2 = min(A[3],B[3])

    if x1 >= x2 or y1 >= y2:
        return 0
    intersection = abs((x1-x2)*(y1-y2))

    union = (A[2]-A[0])*(A[3]-A[1]) + (B[2]-B[0])*(B[3]-B[1]) - intersection

    return intersection/union


def compute_out_dim_conv2d(in_dim,kernel,padding,stride,dilation):
    return int(np.floor(
        ((in_dim+2*padding-dilation*(kernel-1)-1)/stride)+1
    ))

def compute_out_dim_t_conv2d(in_dim,kernel,padding,stride,dilation):
    return int(
        (in_dim-1)*stride-2*padding+dilation*(kernel-1)+1
    )

def draw_ancor_boxes(img,point,sizes,ratios):
    """
    'point' is the central point of the anchors
    """
    img = img.copy()
    x,y = point
    for s in sizes:
        for r in ratios:
            h = s*np.sqrt(r)
            w = s*np.sqrt(1./r)
            bbox = np.array([x-w/2,y-h/2,x+w/2,y+h/2]).astype(int)
            img = draw_bbox(img,bbox,color=255,width=1)
    return img

def get_mean_filter(k):
    return np.ones((k,k))*(1/(k**2))


def get_contrast(img):
    # be sure that the image is in the range [0,1], divide only if the image is in [0,255]
    if np.max(img)>1:
        img = img/255.0
    return np.max(img*255) - np.min(img*255)



def base_stats(img):
    print("Contrasto: {0:.0f} (valore massimo: {1:.0f}; valore minimo: {2:.0f})"
      .format(get_contrast(img),np.max(img*255),np.min(img*255)))




def change_light(img,value=0.5):
    img = (img/255.0)+value
    img = np.clip(img,0,1)
    return img
    
def lower_contrast(img):
    img = (img/255.0)
    return np.clip(img/2,0,1)
def non_linear_lower_contrast(img):
    img = (img/255.0)
    img = np.power(img,1/3)
    img = np.clip(img*1,0,1)
    return img
def invert(img):
    return 1-img

def raise_contrast(img):
    img = (img/255.0)
    return np.clip(img*2,0,1)

def non_linear_raise_contrast(img):
    img = (img/255.0)
    img = np.power(img,2)
    return np.clip(img,0,1)

def binarize(img,theta):
    img = img.copy()
    img[img>=theta]=255
    img[img<theta] = 0
    return img


def show_interpolation(img,bbox):
    """
    'bbox' is the bbox of the analysis' target in the standard fomr [x_min,y_min,x_max,y_max]
    """
    fig = plt.figure(figsize=(30, 30))
    s_x = bbox[0]
    s_y = bbox[1]
    e_x = bbox[2]
    e_y = bbox[3]

    fig.add_subplot(1, 3, 1)
    plt.imshow(ndimage.zoom(img[s_x:e_x,s_y:e_y,0],2,order=0),cmap="gray")
    plt.title("NN")
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.title("Bilinear")
    plt.imshow(ndimage.zoom(img[s_x:e_x,s_y:e_y,0],2,order=1),cmap="gray")
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.title("Bicubic")
    plt.imshow(ndimage.zoom(img[s_x:e_x,s_y:e_y,0],2,order=2),cmap="gray")
    plt.axis('off')

    plt.show()


def image_histogram_equalization(image, number_bins=256):
    """
    s_k = T(r_k) = \sum_j=1^k p_r(r_j) = \sum_j=1^k n_j/n 
    """
    
    n_of_channels = image.shape[2]
    
    image_equalized = np.zeros(image.shape)
    
    for i in range(n_of_channels):
        
        # get image histogram
        image_histogram, bins = np.histogram(image[:,:,i].flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        channel_equalized = np.interp(image[:,:,i].flatten(), bins[:-1], cdf)
        channel_equalized = channel_equalized.reshape(image[:,:,i].shape)
        
        image_equalized[:,:,i] = channel_equalized

    return image_equalized

def my_equalization_gray(img):
    assert img.dtype == np.uint8
    m,n = img.shape[0],img.shape[1]
    if len(img.shape)!=2:
        assert img.shape[2]==1
        img = img.reshape(m,n)

    cumulative = np.cumsum(np.histogram(img.reshape(-1),bins=256)[0])/(m*n)
    img_ret = np.zeros_like(img)
    for i in range(m):
        for j in range(n):
            img_ret[i,j] = np.floor(cumulative[img[i,j]]*255)
    return img_ret

def my_equalization_rgb(img):
    assert len(img.shape)==3
    assert img.shape[2]==3
    m,n = img.shape[0],img.shape[1]
    channels = []
    for i in range(3):
        channels.append(my_equalization_gray(img[:,:,i]).reshape(m,n,1))

    return np.concatenate(channels,axis=2)

def correlation_example(img,bbox,p=0):
    """
    'bbox' is the bbox of the filter, high pixel in the 
    result is related to matching parts of the original 
    image
    """
    filter = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    D = img - np.mean(img)
    F2 = filter - np.mean(filter)
    return  convolution(D,F2,p)


def frequency_filter(img,r,low=True):
    """
    r is the radious of the mask.
    if low is true we have a similar effect to the gaussian filter
    if low is false we detect the border
    """
    im_fft = np.fft.fft2(img)
    img_fft_shited = np.fft.fftshift(im_fft)

    f_abs = abs(im_fft)
    f_bounded = np.log(1+f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    m,n = f_img.shape
    PassCenter = img_fft_shited.copy()
    for i in range(m):
        for j in range(n):
            if low:
                if (i-m//2)**2+(j-n//2)**2 > (r)**2:
                    PassCenter[i,j]*=0
                else:
                    PassCenter[i,j]*=255
            else:
                if (i-m//2)**2+(j-n//2)**2 > (r)**2:
                    PassCenter[i,j]*=255
                else:
                    PassCenter[i,j]*=0
    
    Pass = np.fft.ifftshift(PassCenter)

    inverse_Pass = np.fft.ifft2(Pass)

    ret = np.abs(inverse_Pass)
    return ret


def non_maximal_suppression(img):
    dx = convolution(img,sobelH)
    dy = convolution(img,sobelV)
    magnitude = np.sqrt(dx**2+dy**2)
    img = magnitude
    theta = np.arctan2(dy,dx)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    ret = np.zeros_like(img)

    m,n = ret.shape

    for i in range(1,m-1):
        for j in range(1,n-1):
            # angle 0 and 180
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]
            if (img[i,j] >= q) and (img[i,j] >= r):
                ret[i,j] = img[i,j]
            else:
                ret[i,j] = 0
    ret = (ret-ret.min())/(ret.max()-ret.min())
    ret = (ret*255).astype(int)
    return ret
        
def thresholding(img,theta1,theta2):
    img = img.copy()
    img[img<=theta1] = 0
    img[(theta1<img) & (img<=theta2)] = 128
    img[img>theta2] = 255
    return img
def histeresis(img):
    m,n = img.shape
    img = img.copy()
    for i in range(1,m-1):
        for j in range(1,n-1):
            if img[i,j]!=0 and (255 in img[i-1:i+2,j-1:j+2]):
                img[i,j]=255
            else:
                img[i,j]=0
    return img


def canny(img,low_thresh,high_tresh,sigma):
    img = ndimage.gaussian_filter(img,sigma=sigma)
    img = non_maximal_suppression(img)
    img = thresholding(img,low_thresh,high_tresh)
    img = histeresis(img)
    return img

# need a little fix
def apply_hough_lines(path,num_rhos,num_show,low_thresh,high_tresh,sigma):
    img = read_image_rbg(path).astype(np.uint8)
    img_gray = read_gray_img(path).astype(np.uint8)
    edges = canny(img_gray,low_thresh,high_tresh, sigma).astype(np.uint8)
    lines = cv2.HoughLines(edges, num_rhos, np.pi/180, 200)
    for r_theta in lines[:min(len(lines),num_show)]:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def generate_pyramid(img,sigma,s,num_octaves):
    k = 2**(1/s)
    pyramid = []
    for _ in range(num_octaves):
        octave = [img]
        for _ in range(s+2):
            octave.append(ndimage.gaussian_filter(octave[-1],sigma=k*sigma))
        pyramid.append(octave)
        img = octave[-3][::2,::2]
    return pyramid

def plot_pyramid(p,sz,s,num_octave,hspace=10,vspace=10):
    rows, cols = sz[0],sz[1]

    nrows = sum([x[0].shape[0] for x in p]) +  vspace*(num_octave-1)
    ncols = cols*(s+3)+hspace*(s+2)
    output_image = np.ones((nrows,ncols))

    r = 0
    for i in range(len(p)):
        c = 0
        for j in range(len(p[i])):
            w,h = p[i][j].shape
            output_image[r:r+w,c:c+h] = p[i][j]
            c += cols + hspace
        r += w + vspace
    
    return output_image


# critical choice of sigma
def hough_circle_example(path,num_to_show=1,dp=1.3,minDist=100):
    img_gray = read_gray_img(path)
    img_gray = ndimage.gaussian_filter(img_gray, sigma=3)
    img = read_image_rbg(path)
    cimg = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,dp,minDist)#param1=150,param2=70,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:min(num_to_show,len(circles[0]))]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)
    return cimg,circles




# readable version
def gaussian_kernel(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# more efficient
def gkernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def sign_circles(img,circles):
    """
    signs detected circles with an x
    """
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for t in circles[0]:
        x,y,k = t
        cv2.line(img,(x-k,y-k),(x+k,y+k),(0,255,0),2)
        cv2.line(img,(x-k,y+k),(x+k,y-k),(0,255,0),2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def dilation_example():
    img = np.zeros((10,10))
    img[::2,::2] = 1
    plt.imshow(img)
    plt.title("Original")
    plt.colorbar()
    plt.show()
    
    conv_normal = convolution(img,np.ones((3,3)),p=0)
    plt.imshow(conv_normal)
    plt.title("Normal convolution")
    plt.colorbar()
    plt.show()
    
    k = 3
    filter = np.ones((k,k))
    dilation = 2

    kernel = np.zeros((k*dilation+1,k*dilation+1))
    kernel[::dilation+1,::dilation+1] = filter
    conv_dilation =  convolution(img,kernel,p=0)

    plt.imshow(conv_dilation)
    plt.title("Dilation")
    plt.colorbar()
    plt.show()


# local response normalization
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        """
        Sono possibili due approcci, inter e intra channel.

        Inter channel: the normalization is done in the depth dimension (across the different feature maps):

        .. math::
            b_{x,y}^i = \frac{a_{x,y}^i}{k+\alpha \sum_{j=\max(0,1-n/2)}^{\min(N-1,i+n/2)} (a_{x,y}^j)^2}

        La j varia trai vari canali, mentre la posizione x,y rimane invariata.

        Intra channel: lo spostamnent non è lungo i canali ma nello stesso canale sulle posizioni vicine (rispetto a x,y)
        """
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), 
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0)) 
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
    
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
    


def thresholding_morph(img,theta,low=0,high=1):
    img = img.copy()
    img[img<theta] = low
    img[img>=theta] = high
    return img


# probable bug
def morphologic_thresholding(img,filter,mode,S):
    p = filter.shape[0]//2
    C = convolution(img,filter,p=p)
    m,n = C.shape
    C = C[p:m-p,p:n-p]
    if mode == 'erosion':
        return thresholding_morph(C,S)
    elif mode == 'dilation':
        return thresholding_morph(C,1)
    elif mode == 'opening':
        D = thresholding_morph(C,S)
        return morphologic_thresholding(D,filter,'dilation',S)
    elif mode == 'closing':
        D = thresholding(C,1)
        return morphologic_thresholding(D,filter,'erosion',S-1)
    else:
        raise Exception('mode not supported')
    


def get_sparse_kernel_matrix(K, h_X, w_X):
    # K is the kernel matrix
    # h_X, w_X are the height and width of the input matrix
    # to convolve take the output, let's say W, and the image, let's say X and do:
    # W @ X
    # then you need to reshape the output
    h_K, w_K = K.shape

    h_Y, w_Y = h_X - h_K + 1, w_X - w_K + 1

    W = np.zeros((h_Y * w_Y, h_X * w_X))
    for i in range(h_Y):
        for j in range(w_Y):
            for ii in range(h_K):
                for jj in range(w_K):
                    W[i * w_Y + j, i * w_X + j + ii * w_X + jj] = K[ii, jj]

    return W


def estimate_noise(I):
  """
  Reference: J. Immerkær, “Fast Noise Variance Estimation”, 
  Computer Vision and Image Understanding, 
  Vol. 64, No. 2, pp. 300-302, Sep. 1996 
  """

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(ndimage.convolve(I, M))))
  
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma




def example_harris(img_path,sigma_gauss=None):
    img = cv2.imread(img_path)
    if sigma_gauss is not None and sigma_gauss>0:
        img = ndimage.gaussian_filter(img,sigma_gauss)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    return img
