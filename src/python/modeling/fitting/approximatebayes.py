'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

import algorithm as Algorithm
from ..language import distributions as Dist
from ..language import kernels as Kern
from ..utility import io

import numpy

import dill
import pathos.multiprocessing as mp
from multiprocessing import Lock, Value

import traceback
import inspect
import operator
from types import ModuleType

import time
import copy
import matplotlib.pyplot as plt

from ..utility import plotting
from scipy.stats import norm

''' REJECTION SAMPLING '''

class ABCRejection(Algorithm.ParameterFittingAlgorithm):

	# TODO: don't need a kernel
	@classmethod
	def __call__(cls,task,args={}):

		# Default algorithmic arguments: size of posterior, 0-1 cutoff acceptance
		postSize = 100
		thresh = 0
		acceptFun = lambda task,draw,thresh: task.calculateObjective(draw)<=thresh 
		reportIntervals = [float(i+1)/10 for i in range(11)]

		if 'postSize' in args:
			postSize = args['postSize']
		if 'acceptFun' in args:
			acceptFun = args['acceptFun']
		if 'thresh' in args:
			thresh = args['thresh']

		accepted,weights = [],[]
		iters = 0
		curr = 0

		while len(accepted) < postSize:
			iters += 1
			draw = task.prior.draw()

			p = acceptFun(task,draw,thresh)

			if numpy.random.random() <= p:
				accepted.append(draw)
				weights.append(p) # NOTE: Do we need this? Frequency should account. I think it will lead to underdispersion.

				# Periodically print updates (each time 10% of the posterior is populated)
				if float(len(accepted))/postSize >= reportIntervals[curr]:
					print str(reportIntervals[curr])+" completed"
					curr += 1
					
		print "Acceptance rate: "+str(float(postSize)/iters)

		#return Dist.DiscreteParameterDistribution(accepted,weights)
		return Dist.DiscreteParameterDistribution(accepted)


''' SEQUENTIAL MONTE CARLO (SMC) SAMPLING '''

class ABCSMCAdaptive(Algorithm.ParameterFittingAlgorithm):

	@classmethod
	def __call__(cls,task,args=None):
		
		# Default algorithmic arguments
		postSize = 100
		alpha = 0.5
		cutoff = 0.003
		verbose = True
		outputFile = None
		minErr = 0

		# Alter default arguments if provided
		if args != None:
			if 'postSize' in args.keys():
				postSize = args['postSize']
			if 'alpha' in args.keys():
				alpha = args['alpha']
			if 'cutoff' in args.keys():
				cutoff = args['cutoff']
			if 'verbose' in args.keys():
				verbose = args['verbose']
			if 'outputFile' in args.keys():
				outputFile = args['outputFile']
			if 'minErr' in args.keys():
				minErr = args['minErr']

			# Default 0-1 acceptance kernel (needed by updatePosterior)
			if not 'acceptFun' in args:
				acceptFun = lambda task,draw,thresh: task.calculateObjective(draw)<=thresh 
				args['acceptFun'] = acceptFun
			else:
				acceptFun = args['acceptFun']

			# Default to Gaussian proposal distribution in absence of proposal kernel
			# 'proposalDist' may be provided as a Kernel or a ParameterDistribution
			if not 'proposalDist' in args:
				print "Generating Gaussian kernel from prior samples..."
				kern = Kern.GenerateGaussianKernel(task.prior)
			else:
				# TODO: assert distribution matches prior?
				if isinstance(args['proposalDist'],Kern.Kernel):
					kern = args['proposalDist']
				else:
					kern = Kern.Kernel(args['proposalDist'])

		# Construct initial posterior estimate by rejection sampling 
		if 'acceptFun' in args and 'e0' in args:
			print "Constructing initial posterior estimate by rejection sampling (epsilon = "+str(args['e0'])+")..."
			rejectionSampler = ABCRejection()
			rejectionArgs = {'postSize':postSize,
				'thresh':args['e0'],
				'acceptFun':args['acceptFun']}
			post = rejectionSampler(task,rejectionArgs)
			maxerr = args['e0']
		else:
			print "Constructing initial posterior estimate from prior..."
			post = Dist.DiscreteParameterDistribution([task.prior.draw() for i in range(postSize)])
			errors = [task.calculateObjective(p) for p in post.particles]
			maxerr = max(errors)

		# Set initial error threshold and improvement threshold based on initial
		#  posterior estimate
		args['maxerr'] = maxerr # Needed for Del Moral implementation
		K = maxerr*alpha
		thresh = maxerr-K

		print "Initial Estimate:"
		print post.mean(), post.std()
		if outputFile != None:
			WritePosteriorSummary(outputFile,post,0,maxerr)

		if 'weightFile' in args:
			wts = open(args['weightFile'],"w+")

		# POSTERIOR UPDATE LOOP
		t_spent = 0
		while K > cutoff and maxerr > minErr:
			if verbose:
				print thresh, K, post.mean(), post.ci(0.9)

			t_start = time.time()
			updatedPost = cls.updatePosterior(post,task,kern,thresh,args)
			t_spent += time.time()-t_start

			# Loosen improvement goal in case of failure (max iterations reached)
			if updatedPost == None:
				K = K*(1-alpha)
			else:
				post = updatedPost
				maxerr = thresh
				args['maxerr'] = maxerr # Needed for Del Moral implementation
				K = min(K,maxerr*alpha)

				if outputFile != None:
					WritePosteriorSummary(outputFile,post,t_spent,maxerr)
				if 'weightFile' in args:
					wts.write(str(maxerr)+"\t"+str(numpy.std(post.weights))+"\n") 

			thresh = max(maxerr - K, minErr)

		return post

	@classmethod
	def updatePosterior(cls,post,task,kern,thresh,args):

		postSize = len(post.particles)
		acceptFun = args['acceptFun'] # Required; set in call() method if not provided

		tune = False # NOTE: seems to hurt performance in linear case		
		if 'tune' in args:
			tune = args['tune']

		# Default algorithmic arguments
		maxIters = 5000*postSize # Old implementation terminated after 10k rejections of a single particle update
		parallel = False
		cpus = mp.cpu_count()

		# Alter default arguments if provided
		if args != None:
			if 'maxIters' in args.keys():
				maxIters = args['maxIters']
			if 'parallel' in args.keys():
				parallel = args['parallel']
			if 'cpus' in args.keys():
				cpus = max(2,args['cpus'])

		# Attempt to update each element of the posterior within maxIters 
		updatedParticles, updatedWeights = [], []
		updateParticleTasks = [(post,task,kern,thresh,acceptFun,parallel) for i in range(postSize)]

		if parallel:
			lock = Lock()
			ctr = Value('i',0)
			pool = mp.Pool(cpus,initializer=cls.synchronizationSetup,initargs=(lock,ctr,maxIters))

			updatedParticles = pool.map(cls.updateParticles,updateParticleTasks)
			numIters = ctr.value
			global numIters
		else:
			global numIters
			global maxIters
			numIters = 0
			updatedParticles = map(cls.updateParticles,updateParticleTasks)

		if None in updatedParticles:
			print "Failed"
			return None
		else:
			print "Succeeded in "+str(numIters)+" iterations"

		print "Calculating weights"
		wt_start = time.time() # Print out time spent calculating
		# Calculate weight on accepted particles and store
		if parallel:
			pool = mp.Pool(cpus)
			updatedWeights = pool.map(cls.calculateWeight,[(p,task.prior,post,kern,acceptFun,task,thresh) for p in updatedParticles])
		else:
			updatedWeights = map(cls.calculateWeight,[(p,task.prior,post,kern,acceptFun,task,thresh) for p in updatedParticles])
		wt_dur = time.time()-wt_start
		print "Done with weights ("+str(wt_dur)+" seconds)"

		# Scale kernel for next iteration based on acceptance of current iteration
		# NOTE: Should preemptively scale kernel before sampling current iteration, perhaps based on a fixed number of samples preceding true sampling
		# NOTE: Tuning CANNOT happen between sampling and weight calculation - weight calculation needs access to kernel used to generate samples
		# NOTE: SEEMS TO HURT PERFORMANCE IN LINEAR CASE
		acceptanceRate = float(postSize)/numIters
		if tune:
			print "Tuning ("+str(acceptanceRate)+")"
			cls.setScale(kern,acceptanceRate)

		return Dist.DiscreteParameterDistribution(updatedParticles,updatedWeights)

	# Initialization function for synchronizing between pools
	@classmethod
	def synchronizationSetup(cls,lock,counter,maxval):
		global numItersLock
		global numIters
		global maxIters

		numItersLock = lock
		numIters = counter
		maxIters = maxval

	# Helper function for parallelization
	@classmethod
	def updateParticles(cls,args):
		post = args[0]
		task = args[1]
		kern = args[2]
		thresh = args[3]
		acceptFun = args[4]
		parallel = args[5]

		# IMPORTANT: Workers must exhibit different pseudo-random behavior
		numpy.random.seed()

		draw = None

		while True:
			# Check that maximum iterations have not been exceeded
			# If they have, all remaining workers will exit with 'None'
			if parallel:
				numItersLock.acquire()
				if numIters.value >= maxIters:
					numItersLock.release()
					return None
				numIters.value += 1
				numItersLock.release()
			else:
				if globals()['numIters'] >= globals()['maxIters']:
					return None
				globals()['numIters'] += 1

			x = post.draw()
			draw = kern(x)

			# Kernel perturbation must be valid under prior (respect bounds, etc.)
			if task.prior.pdf(draw) == 0:
				continue

			# Accept the perturbed draw if error criterion is met
			if numpy.random.random() <= acceptFun(task,draw,thresh):
				break

		return draw

	@classmethod
	def calculateWeight(cls,args):
		particle = args[0]
		prior = args[1]
		posterior = args[2]
		kernel = args[3]
		acceptFun = args[4]
		task = args[5]
		thresh = args[6]

		plik = numpy.array([posterior.pdf(p) for p in posterior.particles])
		klik = numpy.array([kernel(p,particle) for p in posterior.particles])

		# NOTE: When I probabilistically accepted AND weighted according to the likelihood, I
		#  found that the posteriors for the linear regression case were underdispersed by a
		#  constant factor. I think you need to either:
		#  - Accept deterministically (or rather with rho > 0) and weight by likelihood, or
		#  - Accept probabilisticallt and weight only on prior/kernel.
		#  The latter case seems to give a smoother distribution, though both seem to match the
		#  expected posterior variance at each iteration.
		#  As such, I've removed the weighting scheme below:
		#  return prior.pdf(particle)*acceptFun(task,particle,thresh) / numpy.sum(plik*klik)
		return prior.pdf(particle) / numpy.sum(plik*klik)

	@classmethod
	def setScale(cls,proposalDist,acceptanceRate):
		scale = proposalDist.scale	

		if acceptanceRate < 0.001:
			proposalDist.scale *= 0.1
		elif acceptanceRate < 0.05:
			proposalDist.scale *= 0.5
		elif acceptanceRate < 0.2:
			proposalDist.scale *= 0.9
		elif acceptanceRate > 0.95:
			proposalDist.scale *= 10.0
		elif acceptanceRate > 0.75:
			proposalDist.scale *= 2.0
		elif acceptanceRate > 0.5:
			proposalDist.scale *= 1.1
		else:
			return False

		# Prevent from tuning to 0
		if proposalDist.scale == 0:
			proposalDist.scale = scale


''' Based on the Del Moral et al. (2011) ABC-SMC algorithm.
	- Samples particles in independent MCMC chains with periodic global resampling
	- O(N) calculation of particle weights
''' 
class ABCSMCDelMoral(ABCSMCAdaptive):

	@classmethod
	def updatePosterior(cls,post,task,kern,thresh,args):

		acceptFun = args['acceptFun']
		prevThresh = args['maxerr']
		nparticles = len(post.particles)

		tune = True		
		if 'tune' in args:
			tune = args['tune']

		# Define the ESS (sum of squared weights) at which resampling will occur
		resampleESS = float(nparticles) * 0.5
		minESS = 0
		if 'minESS' in args:
			minESS = args['minESS'] * float(nparticles)
		if 'resampleESS' in args:
			resampleESS = args['resampleESS'] * float(nparticles)

		# Update the weights in O(N)
		newWeights = []
		for i,p in enumerate(post.particles):
			newWeights.append(post.weights[i])
			if newWeights[i] > 0:
				newWeights[i] *= acceptFun(task,p,thresh)/acceptFun(task,p,prevThresh)

		# Special case of the ESS failure criterion, but prevents divide by 0 errors
		totWt = numpy.sum(newWeights)
		if totWt == 0:
			print "Failed: no surviving particles"
			return None

		newWeights = numpy.array([float(w)/totWt for w in newWeights])
		ESS = 1.0/numpy.sum(newWeights**2)
		print "ESS: "+str(ESS)

		if ESS < minESS:
			print "Failed: below minESS threshold"
			return None

		# Fail if the new threshold would kill more than some critical fraction 
		#  of the population (1-alpha) to prevent point
		# THIS SEEMS TO CAUSE THE SOLVER TO DIE FOR 300 POINTS LINEAR PROBLEM
		#if ESS < 0.1 * nparticles:
		#	print "Failed"
		#	return None

		post = Dist.DiscreteParameterDistribution(post.particles,newWeights)

		# Resample, if necessary
		if ESS < resampleESS:
			print "Resampling..."
			newParticles = [post.draw() for i in range(nparticles)]
			post = Dist.DiscreteParameterDistribution(newParticles)

		# Propose new particles according to a Metropolis-Hastings step from the 
		#   corresponding particle in the previous population.
		newParticles = []
		naccepted, ntries = 0,0
			
		for i,p in enumerate(post.particles):
			# Only sample from "live" particles (or weight the resulting sample to 0)
			if post.pdf(p) == 0:
				newParticles.append(p)
				continue

			proposed = kern.perturb(p)

			num = acceptFun(task,proposed,thresh) * task.prior.pdf(proposed) * kern(proposed,p)
			denom = acceptFun(task,p,thresh) * task.prior.pdf(p) * kern(p,proposed)

			if numpy.random.random() < num/denom:
				newParticles.append(proposed)
				naccepted += 1
			else:
				newParticles.append(p)
			ntries += 1
			acceptanceRate = float(naccepted)/ntries
			
			# Tune MH step to get an ideal acceptance rate every 100 draws
			# NOTE: now that I've fixed scaling and such, perhaps preemptive tuning will work?
			if tune and (i+1)%100 == 0:
				cls.setScale(kern,acceptanceRate)
				naccepted = 0
				ntries = 0

		return Dist.DiscreteParameterDistribution(newParticles,post.weights)


'''
	Helper methods for logging algorithm progress
'''

def WritePosteriorSummary(outputFileHandle,post,t_spent,maxerr):
	io.WriteParameterDistribution(outputFileHandle,post)
