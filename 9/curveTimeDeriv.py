import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime as dt
from scipy.optimize import curve_fit
from util.plot import colorline
from util.seeding import seed
from ctypes import c_double, c_int, Structure, CDLL

class vect(Structure):
    _fields_ = [('x', c_double*10000), 
                ('y', c_double*10000),
                ('z', c_double*10000), 
                ('m', c_double*10000),
                ('hArray', c_double*10000), 
                ('det', c_double*10000), 
                ('its', c_int)]

def squeeze_last(array):
    new_shape = array.shape[:-2]
    new_shape += (array.shape[-1] * array.shape[-2],)
    return np.reshape(array, new_shape)

def create_data(seed_num, iterations, distance, thetas = [0,0], seperations = [1,1], # Simulation Params
                      delta_0 = 10e-6, h0 = 10e-4, safety = .98, ending_tolerance = .1, icity = 1, seed_its = 4, # RKA Params
                      plot = False): 
    
    if len(thetas) != 2 or len(seperations) != 2:
        print('Thetas and seperations must be length 2 arrays')
        return None
    
    # Setup arrays
    thetas = np.linspace(thetas[0], thetas[1], iterations)
    seperations = np.linspace(seperations[0], seperations[1], iterations)
    
    # load C++
    rka_iter = CDLL("./cppScripts/rka_iter_CustT_2Quad").rka_iter_double
    #rka_iter = CDLL("./cppScripts/rka_iter_classical").rka_iter_double
    rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
    rka_iter.restype = vect
    
    # Create seeds
    title, x_seeds, y_seeds, z_seeds = seed(4, distance, seed_num)
    seeds = len(x_seeds)

    # Setup data
    mag = []
    m_first_ar = []
    m_second_ar = []
    curv = []
    c_first_ar = []
    c_second_ar = []
    
    dc_dt = []
    dm_dt = []
    d2c_dt2 = []
    d2m_dt2 = []
    delta_t = (seperations[-1] - seperations[0]) / iterations
    if delta_t == 0:
        delta_t = (thetas[-1] - thetas[0]) / iterations
    if plot:
        fig, ax = plt.subplots(1)
        
    for num, val in enumerate(zip(thetas, seperations)):
        theta, sep = val
        mag_temp = []
        m_first_temp = []
        m_second_temp = []
        curv_temp = []
        c_first_temp = []
        c_second_temp = []
        
        dc_dt_temp = []
        dm_dt_temp = []
        d2c_dt2_temp = []
        d2m_dt2_temp = []
        print(num, val)
        
        for i in range(seeds):
            # Starting point of each field line
            x, y, z = x_seeds[i], y_seeds[i], z_seeds[i]
        
            vect_c = rka_iter(theta*np.pi, sep, 0, x, y, z, seed_its, icity, ending_tolerance, delta_0, safety, h0)
            its = vect_c.its
            x = vect_c.x[0:its]
            y = vect_c.y[0:its]
            z = vect_c.z[0:its]
            m = np.array(vect_c.m[0:its])  
            #print(m)
            #try:
            #    1/0
            #    if m[1] < 0:
            #        continue
            #        mLT0 += 1
            #    else:
            #        mGT0 += 1
                    
            #except:
            #    pass
            #mask = m==0
            #m[mask] = 1
            #m = np.log(m)
    
            if len(x) < seed_its and len(y) < seed_its:
                mag_temp += [np.nan]
                curv_temp += [np.nan]
                if i > 1:
                    m_first_temp += [np.nan]
                    m_second_temp += [np.nan]
                    c_first_temp += [np.nan]
                    c_second_temp += [np.nan]
                    
                    dc_dt_temp += [np.nan]
                    dm_dt_temp += [np.nan]
                    d2c_dt2_temp += [np.nan]
                    d2m_dt2_temp += [np.nan]
                    
                continue
                
            
            x0 = x[0]
            x1 = x[1]
            x2 = x[2]
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            
            Dx = .5 * (x2 - x0)
            D2x = (x0 - 2*x1 + x2)
            Dy = .5 * (y2 - y0)
            D2y = (y0 - 2*y1 + y2)

            # Numerator: (x'y'' - y'x'')
            num = np.abs((Dx * D2y) - (Dy * D2x))
            # Denominator: (x'^2 + y'^2)^(3/2)
            den = ((Dx)**2 + (Dy)**2)**1.5
                        
            curvature = num / den
            m = m[1]
            
            if plot:
                colorline(ax, x[:-1], y[:-1], curvature[:-1], norm = mcolors.LogNorm(vmin=10e-5, vmax=10e5), cmap='jet')
            
            if i > 1:
                c0 = curv_temp[-2]
                c1 = curv_temp[-1]
                c2 = curvature
                m0 = mag_temp[-2]
                m1 = mag_temp[-1]
                m2 = m
                
                m_first = (m0 + m2) / 2
                m_second = (m0 + m1 + m2) / 3
                c_first = (c0 + c2) / 2
                c_second = (c0 + c1 + c2) / 3
                
                if delta_t < 1e-5:
                    dc_dt_val = 0#np.zeros(len(c2))
                    d2c_dt2_val = 0#np.zeros(len(c2))
                    dm_dt_val = 0#np.zeros(len(c2))
                    d2m_dt2_val = 0#np.zeros(len(c2))
                else:
                    # finite difference derivative
                    dc_dt_val = (c2 - c0) / (2*delta_t)
                    dm_dt_val = (m2 - m0) / (2*delta_t)
                    d2c_dt2_val = (c2 - 2*c1 + c0) / (delta_t**2)
                    d2m_dt2_val = (m2 - 2*m1 + m0) / (delta_t**2)
                
                
                mag_temp.append(m2)
                m_first_temp.append(m_first)
                m_second_temp.append(m_second)
                curv_temp.append(c2)
                c_first_temp.append(c_first)
                c_second_temp.append(c_second)
                
                dc_dt_temp.append(dc_dt_val)
                dm_dt_temp.append(dm_dt_val)
                d2c_dt2_temp.append(d2c_dt2_val)
                d2m_dt2_temp.append(d2m_dt2_val)
                
            else:
                mag_temp.append(m)
                curv_temp.append(curvature)
                
        
        mag += [mag_temp]
        m_first_ar += [m_first_temp]
        m_second_ar += [m_second_temp]
        
        curv += [curv_temp]
        c_first_ar += [c_first_temp]
        c_second_ar += [c_second_temp]
        
        dc_dt += [dc_dt_temp]
        dm_dt += [dm_dt_temp]
        d2c_dt2 += [d2c_dt2_temp]
        d2m_dt2 += [d2m_dt2_temp]
        
        
        if plot:
            fig.savefig('theta_%s_sep_%s.png' % (theta, sep))
            fig.clf()
        
    
    # Create np arrays and drop first two indices for non time derivative arrays (Keep same size)
    mag = np.array(mag)
    m_first_ar = np.array(m_first_ar)
    m_second_ar = np.array(m_second_ar)
    
    curv = np.array(curv)
    c_first_ar = np.array(c_first_ar)
    c_second_ar = np.array(c_second_ar)
    
    dc_dt = np.array(dc_dt)
    dm_dt = np.array(dm_dt)
    d2c_dt2 = np.array(d2c_dt2)
    d2m_dt2 = np.array(d2m_dt2)
    
    # Reshape (Index one is time, second is spatial)
    mag = squeeze_last(mag)
    m_first_ar = squeeze_last(m_first_ar)
    m_second_ar= squeeze_last(m_second_ar)
    
    curv = squeeze_last(curv)
    c_first_ar = squeeze_last(c_first_ar)
    c_second_ar= squeeze_last(c_second_ar)
    
    dc_dt = squeeze_last(dc_dt)
    dm_dt = squeeze_last(dm_dt)
    d2c_dt2 = squeeze_last(d2c_dt2)
    d2m_dt2 = squeeze_last(d2m_dt2)
            
    return mag[2:], m_first_ar, m_second_ar, curv[2:], c_first_ar, c_second_ar, dc_dt, dm_dt, d2c_dt2, d2m_dt2

def create_data_no_deriv(seed_num, iterations, distance, thetas = [0,0], seperations = [1,1], # Simulation Params
                      delta_0 = 10e-6, h0 = 10e-4, safety = .98, ending_tolerance = .1, icity = 1, # RKA Params
                      plot = False):
    1/0
    if len(thetas) != 2 or len(seperations) != 2:
        print('Thetas and seperations must be length 2 arrays')
        return None
    
    # Setup arrays
    thetas = np.linspace(thetas[0], thetas[1], iterations)
    seperations = np.linspace(seperations[0], seperations[1], iterations)
    
    # load C++
    rka_iter = CDLL("./cppScripts/rka_iter_custT_2Quad").rka_iter_double
    rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
    rka_iter.restype = vect
    
    # Create seeds
    title, x_seeds, y_seeds, z_seeds = seed(4, distance, seed_num)
    seeds = len(x_seeds)
    seed_its = 3
    
    # Setup data
    mag = []
    curv = []
    r_total = []
    
    if plot:
        fig, ax = plt.subplots(1)
    for num, val in enumerate(zip(thetas, seperations)):
        theta, sep = val
        mag_temp = []
        curv_temp = []
        r_temp = []
        
        print(num, *val)
        
        for i in range(seeds):
            # Starting point of each field line
            x, y, z = x_seeds[i], y_seeds[i], z_seeds[i]
        
            vect_c = rka_iter(theta*np.pi, sep, 0, x, y, z, seed_its, icity, ending_tolerance, delta_0, safety, h0)
            its = vect_c.its
            x = vect_c.x[0:its]
            y = vect_c.y[0:its]
            z = vect_c.z[0:its]
            m = np.array(vect_c.m[0:its])
            m = np.abs(m)
    
            if len(x) < seed_its and len(y) < seed_its:
                mag_temp += [np.nan]
                curv_temp += [np.nan]
                r_temp += [np.nan]
                    
                continue
                            
            x0 = x[0]
            x1 = x[1]
            x2 = x[2]
            y0 = y[0]
            y1 = y[1]
            y2 = y[2]
            
            Dx = .5 * (x2 - x0)
            D2x = (x0 - 2*x1 + x2)
            Dy = .5 * (y2 - y0)
            D2y = (y0 - 2*y1 + y2)

            # Numerator: (x'y'' - y'x'')
            num = np.abs((Dx * D2y) - (Dy * D2x))
            # Denominator: (x'^2 + y'^2)^(3/2)
            den = ((Dx)**2 + (Dy)**2)**1.5
                        
            curvature = num / den
            m = m[1]
            r0 = np.sqrt(x1**2 + y1**2)
            
            if plot:
                colorline(ax, x[:-1], y[:-1], curvature[:-1], norm = mcolors.LogNorm(vmin=10e-5, vmax=10e5), cmap='jet')
            
            mag_temp.append(m)
            curv_temp.append(curvature)
            r_temp.append(r0)
                
        mag += [mag_temp]
        curv += [curv_temp]
        r_total += [r_temp]
        
        if plot:
            ax.set_xlim(-(4/3)*distance, (4/3)*distance)
            ax.set_ylim(-(4/3)*distance, (4/3)*distance)
            
            fig.savefig('visualize_theta_%s_sep_%s.png' % (theta, sep))
            fig.clf()
    
    # Create np arrays and drop first two indices for non time derivative arrays (Keep same size)
    mag = np.array(mag)
    curv = np.array(curv)
    r_total = np.array(r_total)

    # Reshape (Index one is time, second is spatial)
    mag = squeeze_last(mag)
    curv = squeeze_last(curv)
    r_total = squeeze_last(r_total)

    return mag, curv, r_total

def hex_bin_plot(x_array, y_array, x_name='', y_name='', bin_mode = 'log', 
                 x_min=None, x_max = None, y_min=None, y_max=None, gridsize=100, 
                 cmap = 'viridis', poly_order=2, bars = None, ax = None, title = None, save=False):
    
    mask = ~np.isnan(x_array) & ~np.isnan(y_array)
    
    # Find points:
    points = 0
    try:
        for i in x_array:
            if len(i) > points:
                points = len(i)
    except:
        points = 30000 - 1
            
    #plt.figure(figsize=(8,6))
    if ax == None:
        hb = plt.hexbin(
            x_array[mask].ravel(), y_array[mask].ravel(),
            gridsize=gridsize,
            cmap=cmap,
            bins=bin_mode,
            extent=(x_min, x_max, y_min, y_max) if all(v is not None for v in [x_min, x_max, y_min, y_max]) else None
        )
    else:
        hb = ax.hexbin(
            x_array[mask].ravel(), y_array[mask].ravel(),
            gridsize=gridsize,
            cmap=cmap,
            bins=bin_mode,
            extent=(x_min, x_max, y_min, y_max) if all(v is not None for v in [x_min, x_max, y_min, y_max]) else None
        )
    
    counts = hb.get_array()
    #threshold = np.percentile(counts, 90)  # keep upper 20% densest bins
    #print(threshold)
    #mask = counts< threshold
    #print(np.mean(counts))
    counts[counts == 0] = 1   # fill empty bins with 1
    #counts[mask] = 1
    #print(np.mean(counts))
    hb.set_array(counts)
    #hb.set_clim(vmin=1, vmax=1e5)
    
    if bars != None:
        xmin = bars[0]
        xmax = bars[1]
        ymin = bars[2]
        ymax = bars[3]
        plt.hlines(ymin, xmin, xmax)
        plt.hlines(ymax, xmin, xmax)
        plt.vlines(xmin, ymin, ymax)
        plt.vlines(xmax, ymin, ymax)
       
    if bin_mode == 'log':
        plt.colorbar(hb, label='log10 (Counts)')
    else:
        plt.colorbar(hb, label='Linear (Counts)')
        
    if ax == None:
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Density of %s and %s, %s points' % (x_name, y_name, points))
        if title != None:
            plt.title(title)
        plt.tight_layout()
        #plt.yscale('log')
        if save==True:
            plt.title(title)
            plt.savefig(title+'.png')
            plt.clf()
        else:
            plt.show()
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

#Flatten and mask NaNs
def valid(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

def lin(x,a,b):  return a*x + b
def quad(x,a,b,c): return a*x**2 + b*x + c
def expo(x,A,B):  return A*np.exp(B*x)

def fit_and_score(x, y, model, name):
    try:
        p, _ = curve_fit(model, x, y, maxfev=5000)
        yhat = model(x, *p)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        rmse = np.sqrt(np.mean((y - yhat)**2))
        return dict(model=name, params=p, R2=r2, RMSE=rmse)
    except Exception as e:
        return dict(model=name, params=None, R2=np.nan, RMSE=np.nan, error=str(e))

#%% Create data
num_seeds = 1000000 # Seed Points
iterations = 1 # Time steps
distance = 2 # x y spread of seed points
thetas = [0,0] # Start stop of thetas. Total time steps = len(thetas) * len(seps)
#seps = [2.65,2.75] # Start stop of seperatins
seps = [0,0] # Start stop of seperatins
    
start = dt.now()

#mag, curv, r = create_data_no_deriv(num_seeds, 1, distance, thetas, seps, plot=False)
mag, m_first_ar, m_second_ar, curv, c_first_ar, c_second_ar, dc_dt, dm_dt, d2c_dt2, d2m_dt2 = create_data(num_seeds, iterations, distance, thetas, seps, icity = 1, plot=False)


print('Results dict created,', dt.now() - start)


#%%
x = np.linspace(-5.5,10)
plt.plot(x, .213*x - .42)
plt.plot(x, .213*x - .47)

c_test = []
m_test = []
total = 0

for i in range(len(curv)):
    c = np.log(curv[i])
    m = np.log(mag[i])
    
    v1 = .213*m - .42
    v2 = .213*m - .47
    
    if c <= v1 and c >= v2:
        c_test += [c]
        m_test += [m]
        total += 1

c_test = np.array(c_test)
m_test = np.array(m_test)
        
print(total / np.shape(curv)[0])

hex_bin_plot(m_test, c_test, x_name='mag', y_name='Curvature', 
             x_min=-6, x_max = 5, y_min=-2, y_max=1, gridsize=300, bin_mode='log')

#%%
x = np.linspace(-5.5,10)
#plt.plot(x, .213*x - .22)
#plt.plot(x, .213*x - .52)

sep = '2'
theta = '0'

hex_bin_plot(np.log(mag), np.log(curv), x_name='mag', y_name='Curvature (Log)', 
             x_min=-6, x_max = 5, y_min=-10, y_max=5, gridsize=300, bin_mode='log',
             title='Sep: %s Theta: %s' % (sep,theta))
#%%
hex_bin_plot(m_first_ar, np.log(dc_dt), x_name='mag', y_name='Curvature', 
             x_min=0, x_max = 50, y_min=-10, y_max=10, gridsize=300, bin_mode='log')
print(np.nanmax(dm_dt))

#%%
hex_bin_plot(c_first_ar, dm_dt, x_name='mag', y_name='Curvature', 
             x_min=0, x_max = 1.5, y_min=-10, y_max=10, gridsize=300, bin_mode='log')
print(np.nanmax(dm_dt))
print(np.nanmax(dm_dt))
#%%
num_seeds = 1000000 # Seed Points
distance = 6 # x y spread of seed points

thetas_all = np.linspace(0,0,1)
seps_all = np.linspace(0,5,11)

#fig, ax = plt.subplots(nrows = len(thetas_all), ncols=len(seps_all))

for t_num, theta in enumerate(thetas_all):
    for s_num, sep in enumerate(seps_all):
        print(t_num * s_num + s_num)
        thetas = [theta, theta]
        seps = [sep, sep]
        mag, curv, _ = create_data_no_deriv(num_seeds, 1, distance, thetas, seps)
        
        mag[mag==0] = 0.0001
        curv[curv==0] = 0.0001
        
        if np.isnan(np.nanmax(mag)) or np.isnan(np.nanmin(mag)):
            print(t_num, s_num)
            continue
        hex_bin_plot(mag, np.log(curv), x_name='mag', y_name='Curvature (Log)', 
                     x_min=0, x_max = 5, y_min=-2, y_max=1, gridsize=300, bin_mode='log',
                     title='Sep: %s Theta: %s' % (sep,theta),save=True)
        #ax[t_num, s_num].set_title("T: %s S: %s" % (theta, sep))
        #plt.close()
        
        

#%%
num_seeds = 1000000 # Seed Points
iterations = 1 # Time steps
distance = 6 # x y spread of seed points
thetas = [0,0] # Start stop of thetas. Total time steps = len(thetas) * len(seps)
#seps = [2.65,2.75] # Start stop of seperatins
seps = [1,1] # Start stop of seperatins
mag, curv, r = create_data_no_deriv(num_seeds, 1, distance, thetas, seps, plot=False)



        


