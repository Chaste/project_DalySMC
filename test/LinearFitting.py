# Unit testing imports
try:
    import unittest2 as unittest
except ImportError:
    import unittest

from modeling.utility import plotting, io

# Parameter fitting framework imports
from modeling.fitting.objective import LogLikGauss, NegLogLikGauss, SquareError, LogLikUniform
from modeling.fitting.algorithm import ParameterFittingTask
from modeling.fitting.approximatebayes import ABCSMCAdaptive, ABCSMCDelMoral, ABCRejection
from modeling.simulation.experiment import Experiment
import modeling.language.distributions as Dist
from modeling.language.kernels import Kernel

import numpy,time

# Required for quantile-quantile plots
from scipy.stats import norm

sampleSize = 301

class LinearFitting(unittest.TestCase):

	''' Generate data of the form 'y = ax', with iid normal error of known variance added on
		Check that posterior distribution of 'a' follows the analytical normal distribution
		  that is expected when a normal prior is set on 'a'.
	'''
	def _linearSimple(self,objFun,algorithm,algName,algArgs=None,repeats=1):
		a = 5.0				# The true value of 'a'
		sigma_a = 10.0 		# The standard deviation of the prior on 'a'
		sigma_e = 1.0		# The (known) standard deviation of the observation error

		# Generate simulated data ONCE for each set of tests
		if not hasattr(self,'data'):
			print "GENERATING DATA (once)" # Why is this not working?
			self.x = range(-sampleSize/2+1,sampleSize/2+1)
			self.y = numpy.array([a*i for i in self.x])
			self.data = {'output': self.y + numpy.random.normal(0,sigma_e,sampleSize)}
		x = self.x
		y = self.y
		data = self.data

		# Normal prior on 'a', which is conjugate to normal likelihood function
		p_a = Dist.Normal(0.0,sigma_a)
		priors = Dist.IndependentParameterDistribution({'a':p_a})

		# Specify the parameter fitting task
		exp = LinearExperimentSimple()
		task = ParameterFittingTask(priors,exp,data,objFun)

		kern_a = Dist.Normal(0.0,0.1*sigma_a)
		kernel = Kernel(Dist.IndependentParameterDistribution({'a':kern_a}))
		algArgs['proposalDist'] = kernel

		# Specify the analytic form of the posterior on 'a' under Gaussian likelihood on y
		post_mean = numpy.dot(x,self.data['output'])/(numpy.dot(x,x)+1.0/(sigma_a**2))
		post_std = numpy.sqrt(1.0/(numpy.dot(x,x)+1.0/(sigma_a**2)))
		ppf_fun = lambda x: norm.ppf(x,loc=post_mean,scale=post_std)

		print "TRUE MEAN/STD:"
		print post_mean, post_std

		trueValue = {'a':post_mean}

		results = []
		for i in range(repeats):
			# Undo any scaling that may have been applied by MCMC
			kernel.scale = 1.0 

			t_start = time.time()
			results.append(algorithm(task,args=algArgs))
			print "EXECUTION TIME:",algName,time.time()-t_start

			plotting.PlotDiscreteMarginals(results[i],twoDim=False,trueValue=trueValue,
				dest='SimpleLinearMarginals'+algName+str(i),nbins=50)

		plotting.PlotQQ([[p['a'] for p in r.particles] for r in results],
			weights=[r.weights for r in results],ppf=ppf_fun,
			nquantiles=100,dest="SimpleLinearQQplot"+algName)

		return results

	# ABC with a 0-1 acceptance kernel
	def TestLinearSimpleABCToni01(self):
		objFun = SquareError()
		alg = ABCSMCAdaptive()
		alg_args = {"cutoff":1,'minErr':sampleSize,'parallel':True,'postSize':1000,'tune':False,
			'outputFile':'ABCToni01.out'}

		results = self._linearSimple(objFun,alg,"ABCToni01",alg_args)
		print results[0].mean(), results[0].ci(0.9)

	# ABC with a probabilistic acceptance kernel
	# Performance depends highly on normalization constant of acceptFun:
	# - mode:		produces normal distribution with truncated peak
	# - mode*10:	produces normal distribution wider than theoretical,because algorithm 
	#				terminates with eps>1 due to exceeding max iterations
	# - mode*0.1:	produces nearly uniform distribution
	def TestLinearSimpleABCToniProb(self):
		objFun = SquareError()
		alg = ABCSMCAdaptive()

		def acceptFun(task,params,thresh):
			paramCopy = params.copy()
			if 'obj:std' in params:
				paramCopy['obj:std'] = paramCopy['obj:std'] * thresh
			else:
				paramCopy['obj:std'] = 1.0 * thresh
			true = {'a':5.0,'obj:std':paramCopy['obj:std']}
			mode = numpy.exp(-task.calculateObjective(true)/2)
			val = numpy.exp(-task.calculateObjective(paramCopy)/2)/(1.5*mode)
			if val > 1.0:
				print "Modal kernel value exceeds 1.0"
			return min(val,1.0)

		alg_args = {"cutoff":0.1, 'minErr':1, 'parallel':True,'postSize':1000,
			'acceptFun':acceptFun,'e0':1000,'tune':False,'alpha':0.5, 'maxIters':5e6,
			'outputFile':'ABCToniProb.out'}
		results = self._linearSimple(objFun,alg,"ABCToniProb",alg_args)
		print results[0].mean(), results[0].ci(0.9)

	# ABC with a O(N) calculation of weights and a 0-1 acceptance kernel
	def TestLinearSimpleABCDelMoral01(self):
		objFun = SquareError()
		alg = ABCSMCDelMoral()
		alg_args = {"cutoff":1,'minErr':sampleSize,'postSize':1000,'alpha':0.5,
			'outputFile':'DelMoral01.out'}

		results = self._linearSimple(objFun,alg,"ABCDelMoral01",alg_args)
		print results[0].mean(), results[0].ci(0.9)

	# ABC with O(N) calculation of weights and a probabilistic acceptance kernel
	def TestLinearSimpleABCDelMoralProb(self):
		objFun = SquareError()
		alg = ABCSMCDelMoral()

		def acceptFun(task,params,thresh):
			paramCopy = params.copy()
			if 'obj:std' in params:
				paramCopy['obj:std'] = paramCopy['obj:std'] * thresh
			else:
				paramCopy['obj:std'] = 1.0 * thresh
			return min(numpy.exp(-task.calculateObjective(paramCopy)/2),1.0)

		alg_args = {"cutoff":0.001, 'minErr':1, 'postSize':10000, 'acceptFun':acceptFun,
			'e0':1000, 'tune':True, 'alpha':0.5, 
			'outputFile':'DelMoralProb.out'}
		results = self._linearSimple(objFun,alg,"ABCDelMoralProb",alg_args,repeats=1)
		print results[0].mean(), results[0].ci(0.9)

	# Compare uniformity of Del Moral posteriors using 0-1 acceptance over a range of alpha
	def TestDelMoralAlpha(self):
		# Run over a range of alphas
		objFun = SquareError()
		alg = ABCSMCDelMoral()
		alphas = [0.7,0.6,0.5,0.4,0.3,0.2,0.1]
		distrs = []

		for i,alpha in enumerate(alphas):
			alpha_name = str(int(100*alpha))
			outfile_name = 'DelMoral1000_alpha'+alpha_name+'.out'

			alg_args = {"cutoff":1,'minErr':sampleSize,'postSize':1000,'alpha':alpha,
				'outputFile':outfile_name,'tune':True}
			result = self._linearSimple(objFun,alg,"ABCDelMoral01_"+alpha_name,alg_args)
			distrs.append(result[0])

		plotting.PlotMarginalEvolution(distrs,alphas,'alpha',dest='DelMoralAlphaSweep')

'''
	Expects single parameter 'a', returning ax for x in range -sampleSize/2, sampleSize/2
'''
class LinearExperimentSimple(Experiment):
	def simulate(self,parameters):
		results = {'output':[]}
		for i in range(-sampleSize/2+1,sampleSize/2+1):
			results['output'].append(parameters['a']*i)
		return results