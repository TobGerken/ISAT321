import random
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

def calculate_trajectories(start_y,start_z,u,v_sigma,w_sigma,n_walks):
    
    #allocate arrays
    x_coords = np.empty((n_walks,1001))
    y_coords = np.empty((n_walks,1001))
    z_coords = np.empty((n_walks,1001))
    
    for walk in range(0, n_walks):
        position_y = [start_y]
        position_z = [start_z]

        v = np.random.normal(0, v_sigma, 1000)
        w = np.random.normal(0, w_sigma, 1000)

        for vv, ww in zip(v,w):
            position_y.append(position_y[-1]+vv)
            position_z.append(position_z[-1]+ww)   
        
        x_coords[walk,:]=np.arange(0,1001)*u
        y_coords[walk,:]=np.array(position_y)
        z_coords[walk,:]=np.array(position_z)
        
        
    return (x_coords,y_coords,z_coords)


def plot_trajectory(x_coords,y_coords,z_coords,n_trajectories,n_walks):
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(10, 8)
    fig.suptitle('Random Walks')
    
    arr = np.arange(n_walks)
    np.random.shuffle(arr)

    for traj in arr[0:n_trajectories]:
        ax1.plot(x_coords[traj,:], z_coords[traj,:], '-')
        ax1.set_ylabel('Height (m)')
        
        #print(ymin)
        
        ax2.plot(x_coords[traj,:], y_coords[traj,:], '-')
        ax2.set_ylabel('Y-Distance (m)')
        ax2.set_xlabel('X-Distance (m)')
        ymin, ymax = ax2.get_ylim()
        #ax2.vlines(x_coords[1,250],  ymin, ymax ,linestyles='dashed')
        #ax2.vlines(x_coords[1,900],  ymin, ymax ,color='r',linestyles='dashed')
    
    zmin, zmax = ax1.get_ylim()
    ax1.vlines(x_coords[1,250],  zmin, zmax ,linestyles='dashed')
    ax1.vlines(x_coords[1,900],  zmin, zmax ,color='r',linestyles='dashed')
    
    ymin, ymax = ax2.get_ylim()
    ax2.vlines(x_coords[1,250],  ymin, ymax ,linestyles='dashed')
    ax2.vlines(x_coords[1,900],  ymin, ymax ,color='r',linestyles='dashed')
    
    fig, ax3 = plt.subplots(2, 1)
    fig.set_size_inches(10, 8)
    fig.suptitle('Distribution at Crosscuts')
    
    x=250
    ax3[0].set_title('Z-Direction')
    ax3[0].hist(z_coords[arr[0:n_trajectories],x],50,density=True, alpha = 0.5)
    mu, std = norm.fit(z_coords[:,x])
    x = np.linspace(zmin, zmax, 100)
    p = norm.pdf(x, mu, std)
    ax3[0].hlines(norm.pdf(mu+std, mu, std),  mu-std, mu+std ,color='k') 
    ax3[0].plot(x, p, 'b', linewidth=2)
    
    x=900
    ax3[0].hist(z_coords[arr[0:n_trajectories],x],50,density=True,color='r', alpha = 0.5)
    mu, std = norm.fit(z_coords[arr[0:n_trajectories],x])
    x = np.linspace(zmin, zmax, 100)
    p = norm.pdf(x, mu, std)
    ax3[0].hlines(norm.pdf(mu+std, mu, std),  mu-std, mu+std ,color='k')  
    ax3[0].text(mu, norm.pdf(mu+std, mu*1.02, std),  r'$2 \sigma$' ,color='k', 
                verticalalignment='baseline', horizontalalignment='center', fontsize = 12 )  
    ax3[0].plot(x, p, 'r', linewidth=2)
    
    x=250
    ax3[1].set_title('Y-Direction')
    ax3[1].hist(y_coords[arr[0:n_trajectories],x],50,density=True, alpha = 0.5)
    mu, std = norm.fit(y_coords[arr[0:n_trajectories],x])
    x = np.linspace(ymin, ymax, 100)
    p = norm.pdf(x, mu, std)
    ax3[1].plot(x, p, 'b', linewidth=2)
    xmin, xmax = ax3[1].get_xlim()
    #ax3[1].text(xmin,ymax,'Y-coordinate')
    
    
    x=900
    ax3[1].hist(y_coords[arr[0:n_trajectories],x],50,density=True,color='r', alpha = 0.5)
    mu, std = norm.fit(y_coords[arr[0:n_trajectories],x])
    x = np.linspace(ymin, ymax, 100)
    p = norm.pdf(x, mu, std)
    ax3[1].plot(x, p, 'r', linewidth=2)
    ax3[1].legend(['Normal Dist 1','Normal Dist 2','Histogram 1', 'Histogram 2'])


def calc_GaussianPlume(Q,H,u,stability,Param):
    x,y,z,xv, yv, zv, Iy, Iz = Param
    sigma_v = xv*1000 * Iy[stability]
    sigma_z = xv*1000 * Iz[stability]

    #C = Q/u*1/(np.pi*sigma_v *sigma_v)* np.exp(-(yv*1000)**2/(2*sigma_v**2))*np.exp(-H**2/(2*sigma_z**2))
    C = Q/u*1/(np.pi*sigma_v *sigma_v)* np.exp(-(yv*1000)**2/(2*sigma_v**2))*np.exp(-(zv-H)**2/(2*sigma_z**2))

    C=np.where(C < 1e-6, np.nan, C)
    
    return C


def init_GaussianPlume():
    x = np.linspace(0, 40, 400)+.001
    y = np.linspace(-20, 20, 400)+.001
    z = np.linspace(0, 1000, 400)+.001
    Iy = {'A':0.5, 'B':0.3, 'C': 0.15, 'D': 0.1, 'E':0.05}
    Iz = {'A':0.3, 'B':0.12, 'C': 0.07, 'D': 0.05, 'E':0.015}

    xv, yv, zv = np.meshgrid(x, y, z)
    
    NewPlume = True
    return (x,y,z,xv, yv, zv, Iy, Iz),NewPlume


def plot_GaussianPlume(C, XCut, YCut, ZCut, Param):
    x,y,z,xv, yv, zv, Iy, Iz = Param
    
    # translate X,Y,Z to nearest index values 
    X = np.abs(x-XCut)
    ix0 =  np.where( X == X.min() )

    X = np.abs(z-ZCut)
    iz0 =  np.where( X == X.min() )

    X = np.abs(y-YCut)
    iy0 =  np.where( X == X.min() )
    
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 8)

    cf=ax.contourf(x,y,C[:,:,0]*1000)
    cf2=ax.contour(x,y,C[:,:,0]*1000,colors = 'w')
    ax.clabel(cf2, inline=True, fontsize=8)
    ax.set_ylim((-3,3))
    ax.set_xlim((0, 20))
    ax.set_title('Ground Level Concentration')
    ax.set_ylabel('Y-Direction (km)')
    ax.set_xlabel('X-Direction (km)')
    fig.colorbar(cf, ax=ax, label = 'ug/m3')


    fig2, ax = plt.subplots(1, 1)
    fig2.set_size_inches(8, 8)

    ax.plot(x,np.squeeze(C[iy0,:,0]*1000))
    ax.set_xlim((0, 40))
    ax.set_title('Centerline Ground Concentration')
    ax.set_ylabel('ug/m3')


    fig3, ax = plt.subplots(1, 1)
    fig3.set_size_inches(10, 8)

    cf2=ax.contour(x,y,np.squeeze(C[:,:,iz0]*1000),[0.05,0.1,0.5,1,5,10,20,50,100],colors='k')
    ax.clabel(cf2, inline=True, fontsize=8)
    fig.colorbar(cf2, ax=ax, label = 'ug/m3')

    ax.set_xlim((0, 40))
    ax.set_ylim((-3, 3))
    ax.set_title('Concentration Height = %d m' % (ZCut))
    ax.set_xlabel('X-Direction (km)')
    ax.set_ylabel('Y-Direction (km)')

    # Simple contourplot with Y-Z cut
    
    fig4, ax = plt.subplots(1, 1)
    fig4.set_size_inches(10, 8)
    #cf3=ax.contourf(y,z,np.squeeze(C[:,ix0,:]*1000).T)
    cf2=ax.contour(y,z,np.squeeze(C[:,ix0,:]*1000).T)
    fig.colorbar(cf2, ax=ax, label = 'ug/m3')
    ax.clabel(cf2, inline=True, fontsize=8)
    ax.set_xlim((-3, 3))
    ax.set_ylim((0, 500))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('Y-Direction (km)')
    ax.set_title('Concentrations for Crosscut at X = %d km' % (XCut))

    # Simple contourplot with X-Z cut
    fig5, ax = plt.subplots(1, 1)
    fig5.set_size_inches(10, 8)
    cf2=ax.contour(x,z,np.squeeze(C[iy0,:,:].T*1000),[0.05,0.1,0.5,1,5,10,20,50,100])
    fig.colorbar(cf2, ax=ax, label = 'ug/m3')
    ax.clabel(cf2, inline=True, fontsize=8)
    ax.set_xlim((0, 40))
    ax.set_ylim((0, 500))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('X-Direction (km)')
    ax.set_title('Concentrations for Crosscut at Y = %d km' % (YCut))

    return 
