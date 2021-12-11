import numpy as np   

GRID_CHANNELS = 16
MIN_DEEP = 32
MAX_DEEP = 320
SIZE = 256

# UTILS
def random_normal_int(max):
    return int(min(np.abs(np.random.normal(0, 0.3 * (max - 1))), max - 1))


# The main rule for all methods: range of inputs and output values should always be within [-1,1]
methods = {}


def transit_transform  ():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    count = random_normal_int(GRID_CHANNELS) + 1
    s_indx = np.random.choice(GRID_CHANNELS, count, replace=False) 
    alphas = np.abs(np.random.normal(0, 1, count))
    alphas = alphas / np.sum(alphas)
    return 'grid = transit(grid, %s, %s, %s)\n'%(repr(t_indx), repr(list(s_indx)), repr(list(alphas)))
    
methods['transit'] = {
'active': True,
'weight': 1,
'define': """
def transit(x, t_indx, s_indx, alphas):
    res = x.copy()
    res[:,:,t_indx] = np.sum(x[:,:,s_indx] * alphas, axis = -1)
    return test_values(res.clip(-1,1)) 
""",
'transform': transit_transform
}


def sin_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 

    p = np.random.normal(2, 3)
    if p >= 0 : p += 1
    else: p = np.exp(p)
    
    scale = p
    shift = np.random.uniform(-100, 100)
    return 'grid = sin(grid, %s, %s, %s, %s)\n'%(repr(t_indx), repr(s_indx), repr(scale), repr(shift))
   
methods['sin'] = {
'active': True,
'weight': 0.75,
'define': """
def sin(x, t_indx, s_indx, scale = 1, shift = 0): 
    res = x.copy()
    res[:,:,t_indx] = np.sin(x[:,:,s_indx] * 0.5 * np.pi * scale + shift)
    return test_values(res)     
""",
'transform': sin_transform
}


def magnitude_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    count = random_normal_int(GRID_CHANNELS) + 1
    s_indx = np.random.choice(GRID_CHANNELS, count, replace=False) 
   
    p = np.random.normal(0, 0.5)
    if p >= 0 : p += 1
    else: p = np.exp(p)
    
    ord = 2 * p
    return 'grid = magnitude(grid, %s, %s, %s)\n'%(repr(t_indx), repr(list(s_indx)), repr(ord))
   
methods['magnitude'] = {
'active': True,
'weight': 0.45,
'define': """
def magnitude(x, t_indx, s_indx, ord = 2): 
    res = x.copy()
    res[:,:,t_indx] = np.linalg.norm(x[:,:,s_indx], axis = -1, ord = ord) / np.power(len(s_indx), 1 / ord) * 2 - 1
    return test_values(res)   
""",
'transform': magnitude_transform
}

#Provides: shifting image by distorting it
#positive sphere: (1 - abs(((x + 1) / 2) ^ (1 + p) - 1) ^ (1 / (1 + p))) * 2 - 1
#negative sphere:  abs((1 - (x + 1) / 2) ^ (1 + p) - 1) ^ (1 / (1 + p)) * 2 - 1

def shift_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    shift = np.random.normal(0, 3)
    return 'grid = shift(grid, %s, %s, %s)\n'%(repr(t_indx), repr(s_indx), repr(shift))
    
methods['shift'] = {
'active': True,
'weight': 0.35,
'define': """
def shift(x, t_indx, s_indx, shift):
    res = x.copy()
    if shift > 0: res[:,:,t_indx] = (-np.abs(((x[:,:,s_indx] + 1) / 2) ** (1 + shift) - 1) ** (1 / (1 + shift)) + 1) * 2 - 1
    if shift < 0: res[:,:,t_indx] = np.abs((1 - (x[:,:,s_indx] + 1) / 2) ** (1 - shift) - 1) ** (1 / (1 - shift)) * 2 - 1  
    return test_values(res) 
""",
'transform': shift_transform
}


def inverse_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    
    return 'grid = inverse(grid, %s, %s)\n'%(repr(t_indx), repr(s_indx))
    
methods['inverse'] = {
'active': True,
'weight': 0.1,
'define': """
def inverse(x, t_indx, s_indx): 
    res = x.copy()
    res[:,:,t_indx] = -x[:,:,s_indx] 
    return test_values(res)   
""",
'transform': inverse_transform
}



def smooth_max_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s1_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s2_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    return 'grid = smooth_max(grid, %s, %s, %s)\n'%(repr(t_indx), repr(s1_indx), repr(s2_indx))
    
methods['smooth_max'] = {
'active': True,
'weight': 0.2,
'define': """
def smooth_max(x, t_indx, s1_indx, s2_indx, p = 10): 
    res = x.copy()
    res[:,:,t_indx] = np.log((np.exp(x[:,:,s1_indx] * p) + np.exp(x[:,:,s2_indx] * p)) ** (1/p)) / 1.07
    return test_values(res)   
""",
'transform': smooth_max_transform
}


def smooth_min_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s1_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s2_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    return 'grid = smooth_min(grid, %s, %s, %s)\n'%(repr(t_indx), repr(s1_indx), repr(s2_indx))
    
methods['smooth_min'] = {
'active': True,
'weight': 0.2,
'define': """
def smooth_min(x, t_indx, s1_indx, s2_indx, p = 10): 
    res = x.copy()
    res[:,:,t_indx] = -np.log((np.exp(-x[:,:,s1_indx] * p) + np.exp(-x[:,:,s2_indx] * p)) ** (1/p)) / 1.07
    return test_values(res)   
""",
'transform': smooth_min_transform
}

def prod_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    count = random_normal_int(GRID_CHANNELS) + 1
    s_indx = np.random.choice(GRID_CHANNELS, count, replace=False) 
    return 'grid = prod(grid, %s, %s)\n'%(repr(t_indx), repr(list(s_indx)))
    
methods['prod'] = {
'active': True,
'weight': 0.25,
'define': """
def prod(x, t_indx, s_indx):
    res = x.copy()
    res[:,:,t_indx] = np.prod(x[:,:,s_indx], -1)
    return test_values(res) 
""",
'transform': prod_transform
}


def power_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    
    p = np.random.normal(-0.5, 2)
    if p >= 0 : p += 1
    else: p = np.exp(p)
    
    return 'grid = power(grid, %s, %s, %s)\n'%(repr(t_indx), repr(s_indx), repr(p))
    
methods['power'] = {
'active': True,
'weight': 0.25,
'define': """
def power(x, t_indx, s_indx, p = 1): 
    res = x.copy()
    res[:,:,t_indx] = np.sign(x[:,:,s_indx]) * np.abs(x[:,:,s_indx]) ** p 
    return test_values(res)   
""",
'transform': power_transform
}


# INACTIVE #

def shift_mod_abs_transform():
    t_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    s_indx = np.random.choice(GRID_CHANNELS, None, replace=False) 
    shift = np.random.uniform(-1, 1)
    return 'grid = shift_mod_abs(grid, %s, %s, %s)\n'%(repr(t_indx), repr(s_indx),repr(shift))
    
methods['shift_mod_abs'] = {
'active': False,
'weight': 0.1,
'define': """
def shift_mod_abs(x, t_indx, s_indx, shift):
    res = x.copy()
    res[:,:,t_indx] = np.abs(np.mod((x[:,:,s_indx] + 3)/4 + shift/2, 1) * 2 - 1) * 2 - 1
    return test_values(res) 
""",
'transform': shift_mod_abs_transform
}


# DISCARDED #

#This method produce infenetly sharp edges
#Shifting image by repeating it 
def shift_mod_transform():
    count = np.random.randint(1, GRID_CHANNELS + 1)
    indx = np.random.choice(GRID_CHANNELS, count, replace=False) 
    shift = np.random.uniform(-1, 1)
    return 'grid = shift_mod(grid, %s, %s)\n'%(repr(list(indx)),repr(shift))
    
methods['shift_mod'] = {
'active': False,
'weight': 0.1,
'define': """
def shift_mod(x, indx, shift):
    res = x.copy()
    res[:,:,indx] = np.mod((x[:,:,indx] + 1)/2 + shift, 1) * 2 - 1 
    return test_values(res) 
""",
'transform': shift_mod_transform
}


#This method depends on external data
def img_as_func_transform():
    t_indx = np.random.choice(GRID_CHANNELS, 3, replace=False) 
    s_indx = np.random.choice(GRID_CHANNELS, 2, replace=False) 
    
    imgs = ['../img1.jpg', '../img2.jpg', '../img3.jpg', '../img4.jpg', '../img5.jpg', '../img6.jpg']
    img_name = np.random.choice(imgs, None) 
    return 'grid = img_as_func(grid, %s, %s, %s)\n'%(repr(list(t_indx)), repr(list(s_indx)), repr(img_name))
    
methods['img_as_func'] = {
'active': False,
'weight': 2,
'define': """
def img_as_func(x, t_indx, s_indx, img_name): 
    res = x.copy()
    
    img =  Image.open( img_name ).convert('RGB')
    sx = (img.size[1] - 1) / 2
    sy = (img.size[0] - 1) / 2
    img_arr = np.asarray( img, dtype="float32" ) / 127.5 - 1
    
    
    f_x = np.floor((x[:,:,s_indx[0]] + 1) * sx).astype("int32").reshape(x.shape[0] * x.shape[1])
    c_x = np.ceil((x[:,:,s_indx[0]] + 1) * sx).astype("int32").reshape(x.shape[0] * x.shape[1])
    fr_x = (((x[:,:,s_indx[0]] + 1) * sx) % 1).reshape((x.shape[0] * x.shape[1], 1))
    
    f_y = np.floor((x[:,:,s_indx[1]] + 1) * sy).astype("int32").reshape(x.shape[0] * x.shape[1])
    c_y = np.ceil((x[:,:,s_indx[1]] + 1) * sy).astype("int32").reshape(x.shape[0] * x.shape[1])
    fr_y = (((x[:,:,s_indx[1]] + 1) * sy) % 1).reshape((x.shape[0] * x.shape[1], 1))
    
    colors = img_arr[f_x, f_y] * (1 - fr_x) * (1 - fr_y) + \
             img_arr[c_x, f_y] * (    fr_x) * (1 - fr_y) + \
             img_arr[f_x, c_y] * (1 - fr_x) * (    fr_y) + \
             img_arr[c_x, c_y] * (    fr_x) * (    fr_y) 
    colors = colors.clip(-1, 1) # it is necessary because of float rounding errors  
    
    res[:,:,t_indx] = (colors).reshape((x.shape[0], x.shape[1], 3)) 
    return test_values(res)   
""",
'transform': img_as_func_transform
}
