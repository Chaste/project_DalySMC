# Unit testing imports
try:
    import unittest2 as unittest
except ImportError:
    import unittest

from modeling.utility import plotting, io

# Parameter fitting framework imports
from modeling.fitting.objective import SquareError
from modeling.fitting.algorithm import ParameterFittingTask
from modeling.fitting.approximatebayes import ABCSMCAdaptive, ABCSMCDelMoral
from modeling.simulation.experiment import Experiment
import modeling.language.distributions as Dist
from modeling.language.kernels import Kernel

import numpy
sampleSize = 301
xvals = numpy.linspace(-1,1,sampleSize)

class PolynomialFitting(unittest.TestCase):
	'''
		These tests were written to demonstrate the sensitivity of the weighting scheme
		of the Toni SMC/ABC-SMC sampler to increasing dimensionality of the problem.
		- As dimension increases, the variance in the denominator of the weighting scheme
		  increases due to the kernel likelihood term
		- With insufficient particles, this will overwhelm the weighting and collapse the
		  posterior to jagged/point estimates
	'''

	def polynomialFitting(self,objFun,algorithm,algName,algArgs={}):
		exp = PolynomialExperiment()

		all_ndims = [3,6,9,12,16]
		#all_ndims = [16]

		pnames = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
		pvals = [6,-6,5,-5,4,-4,3,-3,2,-2,1,-1,2,-2,3,-3]

		all_results = []

		for ndims in all_ndims:
			print "NUMER OF POLYNOMIAL DIMENSIONS: "+str(ndims)
			
			# Construct the prior and proposal kernel
			priorDists, proposalDists = {}, {}
			trueValues = {}
			for dim in range(ndims):
				priorDists[pnames[dim]] = Dist.Uniform(-10,10)
				proposalDists[pnames[dim]] = Dist.Normal(0,2)
				trueValues[pnames[dim]] = pvals[dim]

			prior = Dist.IndependentParameterDistribution(priorDists)
			proposal = Kernel(Dist.IndependentParameterDistribution(proposalDists))

			# Generate noisy data
			sigma_e = 1.0
			simResults = exp.simulate(trueValues)
			expData = {'output': simResults['output']+numpy.random.normal(loc=0,scale=sigma_e,size=sampleSize)}
			
			task = ParameterFittingTask(prior,exp,expData,objFun)

			outfile = algName+"Poly"+str(ndims)+".out"
			algArgs['outputFile'] = outfile
			wtfile = algName+"Poly"+str(ndims)+".out"
			algArgs['weightFile'] = wtfile

			post = algorithm(task,args=algArgs)
			all_results.append(post)

			plotting.PlotDiscreteMarginals(post,twoDim=True,trueValue=trueValues,
				dest='PolynomialMarginals'+algName+str(ndims),nbins=10)

		for result in all_results:
			print numpy.std(result.weights)

	def TestToni01(self):
		objFun = SquareError()
		alg = ABCSMCAdaptive()
		algArgs = {"cutoff":1,'minErr':sampleSize+1,'parallel':True,'postSize':1000,'tune':False,
			'alpha':0.2}

		self.polynomialFitting(objFun,alg,"Toni",algArgs)

	def TestDelMoral01(self):
		objFun = SquareError()
		alg = ABCSMCDelMoral()
		algArgs = {"cutoff":1,'minErr':sampleSize+1,'postSize':1000,'tune':True,'alpha':0.2,
			'minESS':0.,'resampleESS':0.5}

		self.polynomialFitting(objFun,alg,"DelMoral",algArgs)


class PolynomialExperiment(Experiment):

	def __init__(self):
		self.pnames = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
		self.pvals = [6,-6,5,-5,4,-4,3,-3,2,-2,1,-1,2,-2,3,-3]

	'''	Applies terms to exponential terms of corresponding (decreasing) alphabetical order
		(i.e., 'a'x^0, 'b'x^1, ...)
	'''
	def simulate(self,parameters):
		results = {'output':[]}
		terms = sorted(parameters.keys())

		for x in xvals:
			y = 0
			for i,p in enumerate(self.pnames):
				if p in parameters:
					y += parameters[p] * x**i
				else:
					y += self.pvals[i] * x**i
			results['output'].append(y)

		return results