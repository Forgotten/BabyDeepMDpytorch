import numpy as np

def potential(x,y, mu):
	return -np.exp(-mu*np.abs(y - x))

def forces(x,y, mu):
	return -mu*np.sign(y - x)*np.exp(-mu*np.abs(y - x))

def gen_data(Ncells, Np, mu, Nsamples, minDelta = 0.0, Lcell = 0.0): 

	pointsArray = np.zeros((Nsamples, Np*Ncells))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells))

	for i in range(Nsamples):
		if Lcell == 0.0 :
			sizeCell = 1/Ncells
		else :
			sizeCell = Lcell

		midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

		points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
		points = np.sort(points.reshape((-1,1)), axis = 0)

		# we want to check that the points are not too close 
		while np.min(points[1:] - points[0:-1]) < minDelta:
			points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
			points = np.sort(points.reshape((-1,1)), axis = 0)

		pointsArray[i, :] = points.T

		R = potential(points,points.T, mu)

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = forces(points,points.T, mu)

		Forces = np.sum(F, axis = 1) 

		forcesArray[i,:] = Forces.T

	return pointsArray, potentialArray, forcesArray



def potentialGaussian(x, y, sigma):
	return -np.exp(-sigma*np.square(y - x))


def forcesGaussian(x, y, sigma):
	return -sigma*2*(y - x)*np.exp(-sigma*np.square(y - x))


def gen_dataGaussian(Ncells, Np, sigma, Nsamples, minDelta = 0.0): 

	pointsArray = np.zeros((Nsamples, Np*Ncells))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells))

	for i in range(Nsamples):
		sizeCell = 1/Ncells
		midPoints = np.linspace(sizeCell/2.0,1-sizeCell/2.0, Ncells)

		points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
		points = np.sort(points.reshape((-1,1)), axis = 0)

		# we want to check that the points are not too close 
		while np.min(points[1:] - points[0:-1]) < minDelta:
			points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
			points = np.sort(points.reshape((-1,1)), axis = 0)

		pointsArray[i, :] = points.T

		R = potentialGaussian(points,points.T,sigma)

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = forcesGaussian(points,points.T,sigma)

		Forces = np.sum(F, axis = 1) 

		forcesArray[i,:] = Forces.T

	return pointsArray, potentialArray, forcesArray



def potentialYukawa(x, y, mu):
	return -np.exp(-mu*np.abs(y - x))/np.abs(y - x)


def forcesYukawa(x, y, mu):
	return 	mu*np.sign(y - x)*np.exp(-mu*np.abs(y - x))/np.abs(y - x) + np.sign(y - x)*np.exp(-mu*np.abs(y - x))/np.square(np.abs(y - x))


def genDataYukawa(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

	pointsArray = np.zeros((Nsamples, Np*Ncells))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells))

	for i in range(Nsamples):
		if Lcell == 0.0 :
			sizeCell = 1/Ncells
		else :
			sizeCell = Lcell

		midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

		points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
		points = np.sort(points.reshape((-1,1)), axis = 0)

		# we want to check that the points are not too close 
		while np.min(points[1:] - points[0:-1]) < minDelta:
			points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
			points = np.sort(points.reshape((-1,1)), axis = 0)

		pointsArray[i, :] = points.T

		R = potentialYukawa(points,points.T,sigma)

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = forcesYukawa(points,points.T,sigma)
		F = np.triu(F,1) + np.tril(F,-1)

		Forces = np.sum(F, axis = 1) 

		forcesArray[i,:] = Forces.T

	return pointsArray, potentialArray, forcesArray


def genDataYukawaPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

	pointsArray = np.zeros((Nsamples, Np*Ncells))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells))

	if Lcell == 0.0 :
		sizeCell = 1/Ncells
	else :
		sizeCell = Lcell

	NpointsPerCell = 1000
	Nx = Ncells*NpointsPerCell + 1
	Ls = Ncells*sizeCell

	xGrid, pot, dpotdx = computeDerPotPer(Nx, sigma, Ls)

	idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
	idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
	
	for i in range(Nsamples):

		idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
		idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
		points = xGrid[idxPointCell]
		# this is to keep the periodicity
		pointsExt = np.concatenate([points - Ls, points, points + Ls])
		

		# we want to check that the points are not too close 
		while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
			idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
			idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
			points = xGrid[idxPointCell]
			pointsExt = np.concatenate([points - Ls, points, points + Ls])

		pointsArray[i, :] = points.T

		R = pot[idxPointCell - idxPointCell.T]

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = dpotdx[idxPointCell - idxPointCell.T]
		F = np.triu(F,1) + np.tril(F,-1)

		Forces = -np.sum(F, axis = 1) 

		forcesArray[i,:] = Forces.T

	return pointsArray, potentialArray, forcesArray


def gaussian(x, xCenter, tau):
	return (1/np.sqrt(2*np.pi*tau**2))*\
		   np.exp( -0.5*np.square(x - xCenter)/tau**2 )

def gaussianNUFFT(x, xCenter, tau):
	return np.exp(-np.square(x - xCenter)/(4*tau) )

def gaussianDeconv(k, tau):
	return np.sqrt(np.pi/tau)*np.exp( np.square(k)*tau)

def computeDerPotPer(Nx, mu, Ls, xCenter = 0, nPointSmear = 10):   
	
	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx)) 
	mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

	# here we smear the dirac delta
	# we use the width of the smearing for 
	tau = nPointSmear*Ls/Nx

	x = gaussian(xGrid, xCenter, tau) + \
		gaussian(xGrid - Ls, xCenter, tau) +\
		gaussian(xGrid + Ls, xCenter, tau) 

	xFFT = np.fft.fftshift(np.fft.fft(x))
	
	yFFT = xFFT*mult
	
	y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

	dydxFFT = 1.j*kGrid*yFFT
	dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

	return xGrid, y, np.real(dydx)

def computeLJ_FPotPer(Nx, mu, Ls, cutIdx = 50, xCenter = 0, nPointSmear = 10):
	
	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx)) 
	mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

	# here we smear the dirac delta
	# we use the width of the smearing for 
	tau = nPointSmear*Ls/Nx

	x = gaussian(xGrid, xCenter, tau) + \
		gaussian(xGrid - Ls, xCenter, tau) +\
		gaussian(xGrid + Ls, xCenter, tau) 

	xFFT = np.fft.fftshift(np.fft.fft(x))
	
	yFFT = xFFT*mult
	
	y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

	dydxFFT = 1.j*kGrid*yFFT
	dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

	potPer = potentialLJ(xGrid,    xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
			 potentialLJ(xGrid-Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
			 potentialLJ(xGrid+Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx])   

	derPotPer = forcesLJ(xGrid,    xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
				forcesLJ(xGrid-Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
				forcesLJ(xGrid+Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx])   

	y = y + potPer
	dydx = np.real(dydx) + derPotPer

	return xGrid, y, dydx


def potentialLJ(x,y, epsilon, sigma):
	return 4*epsilon*(pow(sigma/np.abs(x-y), 12) - pow(sigma/np.abs(x-y), 6) )

def forcesLJ(x,y, epsilon, sigma):
	return 4*epsilon*(-12*sigma*np.sign(x-y)*pow(sigma/np.abs(x-y), 13) + \
						6*sigma*np.sign(x-y)*pow(sigma/np.abs(x-y), 7) )

def genDataLJ_FPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

	pointsArray = np.zeros((Nsamples, Np*Ncells))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells))

	if Lcell == 0.0 :
		sizeCell = 1/Ncells
	else :
		sizeCell = Lcell

	NpointsPerCell = 1000
	Nx = Ncells*NpointsPerCell + 1
	Ls = Ncells*sizeCell
	cutIdx = round(Nx*minDelta/Ls)

	xGrid, pot, dpotdx = computeLJ_FPotPer(Nx, sigma, Ls, cutIdx = cutIdx)

	# this is the old version 
	# idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)

	idxCell = np.linspace(cutIdx//2,NpointsPerCell-1-cutIdx//2, NpointsPerCell- 2*(cutIdx//2)).astype(int)
	idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
	
	for i in range(Nsamples):

		idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
		idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
		points = xGrid[idxPointCell]
		# this is to keep the periodicity
		pointsExt = np.concatenate([points - Ls, points, points + Ls])
		

		# we want to check that the points are not too close 
		while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
			idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
			idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
			points = xGrid[idxPointCell]
			pointsExt = np.concatenate([points - Ls, points, points + Ls])

		pointsArray[i, :] = points.T

		R = pot[idxPointCell - idxPointCell.T]

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = dpotdx[idxPointCell - idxPointCell.T]
		F = np.triu(F,1) + np.tril(F,-1)

		Forces = -np.sum(F, axis = 1) 

		forcesArray[i,:] = Forces.T

	return pointsArray, potentialArray, forcesArray



# this doesn't really work due to the Gibbs phenomenon. 
def computeDerPotPerNUFFT(Nx, mu, Ls, xCenter = 0):
	
	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx)) 
	mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

	# here we smear the dirac delta
	# we use the width of the smearing for 
	tau = 12*(Ls/(2*np.pi*Nx))**2 

	x = gaussianNUFFT(xGrid, xCenter, tau) + \
		gaussianNUFFT(xGrid - Ls, xCenter, tau) +\
		gaussianNUFFT(xGrid + Ls, xCenter, tau) 

	xFFT = np.fft.fftshift(np.fft.fft(x))

	filterDeconv = gaussianDeconv(kGrid, tau)

	xFFTDeconv = xFFT*filterDeconv
	yFFT = xFFTDeconv*mult/(2*np.pi*Nx/Ls)
	
	y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

	dydxFFT = 1.j*kGrid*yFFT
	dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

	return xGrid, y, np.real(dydx)
