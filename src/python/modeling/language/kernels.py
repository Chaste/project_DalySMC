'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

from distributions import IndependentParameterDistribution, Normal

''' Kernels will be handled as a wrapper class around ParameterDistribution objects.
	Customization of kernels beyond collections of standard univariate distributions
	  are currently supposed to be handled by custom implementations of ParameterDistribution.
'''
class Kernel:
	''' Accepts a ParameterDistribution object that is used to perturb kernels or calculate
		the likelihood of a provided perturbation.
		Optional scale parameter controls width of step uniformly across all parameters.
	'''
	def __init__(self,parameterDistribution,scale=1.0):
		self.parameterDistribution = parameterDistribution
		self._scale = scale

	def __repr__(self):
		return str(self.parameterDistribution)

	@property
	def scale(self):
		return self._scale
	@scale.setter
	def scale(self,value):
		self._scale = value
	
	''' When provided with a single argument, returns a dictionary of perturbed values
		When provided with two arguments, returns the probability of reaching one set of
		  parameters from the other under a single kernel step. 
	'''
	def __call__(self,curr,next=None):
		if next == None:
			return self.perturb(curr)
		else:
			return self.kpdf(curr,next)

	''' Returns a new dictionary of 'values' perturbed accoring to the ParameterDistribution.
		  Does NOT alter original values. 
		Perturbs ONLY entries of 'values' specified in the ParameterDistribution. This allows
		  for partial alteration of a parameter set.
	'''
	def perturb(self,values):
		perturbation = self.parameterDistribution.draw()
		result = values.copy()

		for key,delta in perturbation.iteritems():
			result[key] = result[key]+(delta*self.scale)
		return result

	''' Calculates the probability of generating 'next' from 'curr' under the kernel 
		Assumes bijection between names in ParameterDistribution and curr/next
		NOTE: IndependentParameterDistribution pdf() does not assume bijection; could support
		  kernels operating on only a subset of parameters
	'''
	def kpdf(self,curr,next):
		# NOTE: will allocation of a dictionary on each call to kpdf be too costly?
		# Should we modify an internal 'perturbation' object for applications like ABC?
		perturbation = dict([(key,(next[key]-val)/self.scale) for key,val in curr.iteritems()])
		return self.parameterDistribution.pdf(perturbation)


''' Helper method for MCMC/ABC methods when the user does not specify a proposal distribution
	Constructs a independent Gaussian kernel, with marginal standard deviations equal to
	  10% of the range of the parameter observed across 100 samples.
	Accepts:	
		- dist: ParameterDistribution from which to sample
'''
def GenerateGaussianKernel(dist):
	# Construct a dictionary of (min,max) pairs for each parameter
	bounds = dict([(key,[val,val]) for key,val in dist.draw().iteritems()])

	for i in range(100):
		draw = dist.draw()
		for pname,val in draw.iteritems():
			if val < bounds[pname][0]:
				bounds[pname][0] = val
			if val > bounds[pname][1]:
				bounds[pname][1] = val

	kdist = dict([(pname,Normal(0,0.1*(val[1]-val[0]))) for pname,val in bounds.iteritems()])
	return Kernel(IndependentParameterDistribution(kdist))