import numpy as np


def potential(x,y, mu):

    return -np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1)))

def potential_diff(diff, mu):

    return -np.exp(-mu*np.sqrt(np.sum(np.square(diff), axis = -1)))

def forces(x,y, mu):

    return -mu*(y - x)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(y - x), \
                                                           axis = -1, keepdims = True)))\
           *np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1, keepdims = True)))

def forces_diff(diff, mu):

    return mu*(diff)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(diff), \
                                        axis = -1, keepdims = True)))\
           *np.exp(-mu*np.sqrt(np.sum(np.square(diff), \
                                      axis = -1, keepdims = True)))


def potential_per_3d(x,y, mu, L):
    diff = y - x
    diff_per = diff - L*np.round(diff/L)
    return    potential_diff(diff_per, mu)


def forces_per3D(x,y, mu, L):
    diff = y - x
    diff_per = diff - L*np.round(diff/L)
    return    forces_diff(diff_per, mu)


def gaussian3D(x, y, z, center, tau):

    return (1/np.sqrt(2*np.pi*tau)**3)*\
           np.exp( -0.5*(  np.square(x - center[0]) \
                         + np.square(y - center[1]) \
                         + np.square(z - center[2]))/tau**2 )



def gen_data_per_3d_mixed(n_cells, Np, mu1, mu2, 
                          n_samples, minDelta, Lcell, 
                          weight1,weight2): 
    

    pointsArray = np.zeros((n_samples, Np*n_cells**3, 3))
    potentialArray = np.zeros((n_samples,1))
    forcesArray = np.zeros((n_samples, Np*n_cells**3, 3))


    sizeCell = Lcell
    L = sizeCell*n_cells

    # define a mesh
    midPoints = np.linspace(sizeCell/2.0,n_cells*sizeCell-sizeCell/2.0, n_cells)
    yy, xx, zz = np.meshgrid(midPoints, midPoints, midPoints)
    midPoints = np.concatenate([np.reshape(xx, (n_cells, n_cells, 
                                                n_cells, 1,1)), 
                                np.reshape(yy, (n_cells, n_cells, 
                                                n_cells, 1,1)),
                                np.reshape(zz, (n_cells, n_cells, 
                                                n_cells, 1,1))], axis = -1) 

    # loop over the samples
    for i in range(n_samples):

        # generate the index randomly
        points = midPoints + sizeCell*(np.random.rand(n_cells, n_cells, 
                                                      n_cells, Np, 3)-0.5)
        relPoints = np.reshape(points, (-1,1,3)) -np.reshape(points, (1,-1,3))
        relPointsPer = relPoints - L*np.round(relPoints/L)
        distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))

        # to avoid two points are too close: sigularity
        while np.min( distPoints[distPoints>0] ) < minDelta:

            points = midPoints + sizeCell*(np.random.rand(n_cells, n_cells, 
                                                          n_cells, Np, 3)-0.5)
            relPoints = np.reshape(points, (-1,1,3)) -np.reshape(points, (1,-1,3))    
            relPointsPer = relPoints - L*np.round(relPoints/L)
            distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))


        # compute the points,weighted energy and weighte force
        pointsArray[i, :, :] = np.reshape(points,(Np*n_cells**3, 3))
        points  = np.reshape(points, (Np*n_cells**3, 1, 3))
        pointsT = np.reshape(points, (1, Np*n_cells**3, 3))

        R1 = potential_per_3d(points, pointsT, mu1, L)
        RR1 = np.triu(R1, 1)
        potTotal1 = np.sum(RR1)

        R2 = potential_per_3d(points, pointsT, mu2, L)
        RR2 = np.triu(R2, 1)
        potTotal2 = np.sum(RR2)

        potentialArray[i,:] = potTotal1*weight1 + potTotal2*weight2


        F1 = forces_per3D(points, pointsT, mu1, L)
        Forces1 = np.sum(F1, axis = 1) 

        F2 = forces_per3D(points, pointsT, mu2, L)
        Forces2 = np.sum(F2, axis = 1) 

        forcesArray[i,:,:] = np.reshape(Forces1,(Np*n_cells**3, 3))*weight1 +\
                             np.reshape(Forces2,(Np*n_cells**3, 3))*weight2

    return pointsArray, potentialArray, forcesArray


####################################################################
####################################################################

# Exponential type data for different charges

def gen_data_exp_per_3d_charge(Ncells=10, Np=1, Nsamples=10, 
                               Lcell=1.0, minDelta=0.5, 
                               mindipole = 0.1, maxdipole = 0.1,
                               min_theta = 0., max_theta = np.pi,
                               min_phi = 0., max_phi = 2*np.pi, mu = 10.0):
    ##Ncells : number of cells
    ##Np: number of dipoles in each cell
    ##Nsamples: 
    ##Lcell: length of cell
    ##minDelta: min distance between two dioples centroid
    ##maxdipole: max distance between two particles in a dipoles 
    ##(min_theta, max_theta): range for sampling theta
    ##(min_phi, max_phi): range for sampling phi   
    pointsArray = np.zeros((Nsamples, 2*Np*Ncells**3,3))
    chargeArray = np.zeros((Nsamples, 2*Np*Ncells**3,1))
    potentialArray = np.zeros((Nsamples,1))
    forceArray = np.zeros((Nsamples, 2*Np*Ncells**3,3))

    Ncenter = Np*Ncells**3
    sizeCell = Lcell 
    L = sizeCell*Ncells

    midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)
    yy, xx, zz = np.meshgrid(midPoints, midPoints, midPoints)
    midPoints = np.concatenate([np.reshape(xx, (Ncells, Ncells, 
                                                Ncells, 1,1)), 
                                np.reshape(yy, (Ncells, Ncells, 
                                                Ncells, 1,1)),
                                np.reshape(zz, (Ncells, Ncells, 
                                                Ncells, 1,1))], axis = -1)
    ## (Ncells, Ncells, Ncells, 1, 3)     
    for i in range(Nsamples):
        points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, 
                                                      Ncells, Np, 3) - 0.5)
        ## (Ncells, Ncells, Ncells, Np, 3)         
        relPoints = np.reshape(points, (-1,1,3)) - np.reshape(points, (1,-1,3))
        relPointsPer = relPoints - L*np.round(relPoints/L)
        distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))
        
        # repeating the choice of points until they are minDelta
        # away from each other
        while np.min(distPoints[distPoints>0]) < minDelta:
            points = midPoints + sizeCell*(np.random.rand(Ncells,
                                                          Ncells, 
                                                          Ncells, Np, 3) - 0.5)
            ## (Ncells, Ncells, Ncells, Np, 3)         
            relPoints = np.reshape(points, (-1,1,3))\
                      - np.reshape(points, (1,-1,3))
            relPointsPer = relPoints - L*np.round(relPoints/L)
            distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))
                  

        # sampling the distance of the dipoles
        rlist = np.random.uniform(low=mindipole, high=maxdipole,
                                  size=(Ncells*Ncells*Ncells*Np, 1))

        # sampling the Euler angles 
        phi = np.random.uniform(low=min_phi, high=max_phi, 
                                size=(Ncells*Ncells*Ncells*Np, 1))

        theta = np.random.uniform(low=min_theta, high=max_theta, 
                                  size=(Ncells*Ncells*Ncells*Np, 1))        
        
        # buidlfing the positions of the points based the angles 
        routput = np.concatenate([rlist*np.sin(theta)*np.cos(phi), 
                                  rlist*np.sin(theta)*np.sin(phi),
                                  rlist*np.cos(theta)], axis=-1)

        ## (Ncells*Ncells*Ncells*Np,3)
        pointsreshape = np.reshape(points,(Np*Ncells**3,3)) 

        # creating the point by adding and substracting from the itial point       
        pointsparticle = np.concatenate([np.reshape(pointsreshape - routput,
                                                    (Np*Ncells**3,1,3)), 
                                         np.reshape(pointsreshape + routput,
                                                    (Np*Ncells**3,1,3))],axis=1)
        ## (Ncells*Ncells*Ncells*Np,2,3)
        pointsparticle = np.reshape(pointsparticle,(2*Ncells*Ncells*Ncells*Np,3))
        ## (2*Ncells*Ncells*Ncells*Np,3)
        ## position of each particle

        # We create the charges, in this case we only have dipoles 
        chargelist = []
        for ii in range(Ncenter):
            chargelist.append([-1.0,1.0])
        chargelist = np.array(chargelist).reshape((-1,1))
        
        ###check shape
        pointsArray[i, :] = pointsparticle
        chargeArray[i, :] = chargelist
        
        # computing the energy
        Energy = potential_exp_per_3d_charge(pointsparticle,
                                             chargelist,
                                             L, mu)
        # anc computing the forces using the dipole description
        Force = force_exp_per_sep_3d_charge(pointsparticle,
                                            chargelist,
                                            L, mu)

        # Storing the potential and forces in the arrays
        potentialArray[i,:] = Energy
        forceArray[i,:] = Force

    return pointsArray, chargeArray, potentialArray, forceArray


def potential_exp_per_3d_charge(points, charge, L, mu=10.0):
    ##points: (2*Ncenter,3) position of the points 
    ##charge: (2*Ncenter,1)

    n_points = points.shape[0]

    points1 = np.reshape(points,(n_points,1,3))
    pointsT = np.reshape(points,(1,n_points,3))

    charge1 = np.reshape(charge,(n_points,1))
    chargeT = np.reshape(charge,(1,n_points))
    
    points_diff = points1-pointsT
    charge_mult = charge1*chargeT

    points_diff_per = points_diff - L*np.round(points_diff/L)

    Energyall = charge_mult*potential_diff(points_diff_per, mu)
    Energyall1 = np.triu(Energyall, 1)

    Energy = np.sum(Energyall1)

    return Energy

def force_exp_per_sep_3d_charge(points, charge, L, mu=10.0):
    ## the gradient with respect to x
    ## x can be a value, y can be vector
    ## x_moment : (1,3); x_center : (1,3); x_charge : ()
    ## y_moment : (:,3); y_center : (:,3)
    n_points = points.shape[0]

    points1 = np.reshape(points,(n_points,1,3))
    pointsT = np.reshape(points,(1,n_points,3))

    charge1 = np.reshape(charge,(n_points,1,1))
    chargeT = np.reshape(charge,(1,n_points,1))
    
    points_diff = points1-pointsT
    charge_mult = charge1*chargeT

    points_diff_per = points_diff - L*np.round(points_diff/L)

    Force_all = charge_mult*forces_diff(points_diff_per, mu)

    F= np.sum(Force_all, axis = 1) 

    return F


