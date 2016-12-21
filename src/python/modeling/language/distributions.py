'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

from abc import ABCMeta, abstractmethod
import scipy.stats as stats
import numpy
import numpy.random as rand

'''
	ABSTRACT CLASSES:
		Distribution
		ParameterDistribution 
'''

''' Base class for univariate distributions over single model parameters '''
class Distribution:
	__metaclass__ = ABCMeta

	''' Returns value randomly drawn from the distribution '''
	@abstractmethod
	def draw(self):	pass

	''' Takes float value and returns float probability '''
	# NOTE: despite naming convention, can support discrete distributions
	@abstractmethod
	def pdf(self,value): pass

	''' Takes float value and returns float probability '''
	# NOTE: despite naming convention, can support discrete distributions
	@abstractmethod
	def cdf(self,value): pass


''' Base class for distributions over model parameter vectors '''
class ParameterDistribution:
	__metaclass__ = ABCMeta

	''' Returns a dict of {name,value} pairs representing a random draw from the distribution '''
	@abstractmethod
	def draw(self): pass

	''' Accepts a dict of {name,value} pairs and returns the probability under the distribution '''
	# NOTE: despite naming convention, can support discrete distributions
	@abstractmethod
	def pdf(self,values): pass

	''' Accepts a dict of {name,value} pairs and fixes the values of the specified parameters '''
	''' Required only for partial/sequential model fitting (marginalize distribution) '''
	def fix(self,fixedParameters):
		raise NotImplementedError


''' 
	IMPLEMENTED DISTRIBUTIONS:
		Uniform
		Normal
		Arbitrary
		Point
'''
# TODO: Change all naming conventions to protect variables (leading underscores)
class Uniform(Distribution):
	''' Initializes a uniform distribution between 'low' and 'high' '''
	def __init__(self,low=0.0,high=1.0):
		self.low = float(min(low,high))
		self.high = float(max(low,high))
	def __repr__(self):
		return "".join(["Uniform(",str(self.low),",",str(self.high),")"])

	def draw(self):
		return rand.uniform(self.low,self.high)
	def pdf(self,value):
		u = stats.uniform(loc=self.low,scale=self.high-self.low)
		return u.pdf(value)
	def cdf(self,value):
		u = stats.uniform(loc=self.low,scale=self.high-self.low)
		return u.cdf(value)

	# NOTE: Some algorithms (OLS) check for 'min' and 'max' properties to set bounds.
	# Custom distributions with bounds should respect this naming convention.
	def min(self):
		return self.low
	def max(self):
		return self.high

	def mean(self):
		return (self.high-self.low)/2
	def std(self):
		return 1.0/numpy.sqrt(12)*(self.high-self.low)

class Normal(Distribution):
	''' Initialize a normal distribution with mean 'mean' and standard deviation 'std' '''
	def __init__(self,mean=0.0,std=1.0):
		self.__mean = float(mean)
		self.__std = float(std)
	def __repr__(self):
		return "".join(["Normal(",str(self.mean()),",",str(self.std()),")"])

	def draw(self):
		return rand.normal(self.mean(),self.std())
	def pdf(self,value):
		N = stats.norm(loc=self.mean(),scale=self.std())
		return N.pdf(value)
	def cdf(self,value):
		N = stats.norm(loc=self.mean(),scale=self.std())
		return N.cdf(value)

	def mean(self):
		return self.__mean
	def std(self):
		return self.__std


class Arbitrary(Distribution):
	''' Initializes a discrete distribution across 'vals' with optional 'weights' '''
	def __init__(self,values,weights=None):
		self.values = values
		if weights == None:
			self.weights = numpy.ones(len(values))/len(values)
		else:
			# Normalize weights (may require float casting for ints)
			self.weights = numpy.asarray([float(w) for w in weights])
			self.weights = self.weights/sum(weights)
	# NOTE: duplicate values will not be removed, but Distribution methods won't be affected
	def __repr__(self):
		return "Arbitrary("+str(self.values)+","+str(self.weights)+")"

	# NOTE: O(N) implementations could be sped with pre-processing
	def draw(self):
		r,curr = rand.rand(), 0.0
		for i,w in enumerate(self.weights):
			curr = curr+w
			if curr >= r:
				return self.values[i]
	def pdf(self,value):
		# This approach is used instead of index() due to the possibility of duplicates
		indices = [i for i,x in enumerate(self.values) if x==value]
		return sum([self.weights[i] for i in indices])
	def cdf(self,value):
		indices = [i for i,x in enumerate(self.values) if x<=value]
		# Catch floating point rounding error (weights may sum to <1.0)
		if len(indices) == len(self.values):
			return 1.0
		return sum([self.weights[i] for i in indices])

	def min(self):
		return min(self.values)
	def max(self):
		return max(self.values)

class Point(Distribution):
	''' Initializes an instantaneous distribution for a parameter with a fixed value '''
	def __init__(self,value):
		self.value = value
	def __repr__(self):
		return str(self.value)

	def draw(self):
		return self.value
	def pdf(self,value):
		return float(self.value==value)
	def cdf(self,value):
		return float(self.value<=value)

	def min(self):
		return self.value
	def max(self):
		return self.value


'''
	IMPLEMENTED PARAMETER DISTRIBUTIONS:
		IndependentParameterDistribution
		DiscreteParameterDistribution
'''

class IndependentParameterDistribution(ParameterDistribution):
	''' Handles parameter vector distribution as a dictionary of Distribution objects
			distributions: {name,Distribution}
			constraints: {'eq'|'ineq',<callable>}, or sequence of said dicts. The callable
				object/function accepts a parameter list and returns 0 to pass an equality
				constraint or a non-negative value to pass an inequality constraint.
	'''
	def __init__(self,distributions,constraints=None):
		self.distributions = distributions
		# NOTE: Should we check constraints for valid formatting?
		self.constraints = constraints
	def __repr__(self):
		return ", ".join([key+"~"+str(dist) for key,dist in self.distributions.iteritems()])

	def checkConstraints(self,values):
		if self.constraints == None:
			return True
		# NOTE: because of specification, if 'type' is incorrectly provided, will default
		# to the more general inequality constraint.
		if isinstance(self.constraints,list):
			for const in self.constraints:
				if not const(values):
					return False
			return True
		else:
			return self.constraints(values)

	def draw(self):
		ret = dict([(key,dist.draw()) for key,dist in self.distributions.iteritems()])
		if self.checkConstraints(ret):
			return ret
		else:
			# NOTE: should we put an iteration limit (or a warning) for when constraints
			# might be impossible/too strict? Is there a way to test for this a priori?
			while True:
				# Mutates existing dictionary to save allocation time
				for key,dist in self.distributions.iteritems():
					ret[key] = dist.draw()
				if self.checkConstraints(ret):
					return ret

	def pdf(self,values):
		prob = 1.0
		if not self.checkConstraints(values):
			return 0.0
		else:
			# NOTE: does not enforce bijection between keys. 
			#       'values' dictionary could contain irrelevant keys.
			for key, dist in self.distributions.iteritems():
				prob = prob * dist.pdf(values[key])
			return prob

	# TODO: Constraints are defined by rejection sampling, so the min/max values
	#  may not necessarily respect them.
	def min(self):
		mins = {}
		for key,dist in self.distributions.iteritems():
			if hasattr(dist,'min'):
				mins[key] = dist.min()
			else:
				mins[key] = -float('inf')
		return mins
	def max(self):
		maxs = {}
		for key,dist in self.distributions.iteritems():
			if hasattr(dist,'max'):
				maxs[key] = dist.max()
			else:
				maxs[key] = float('inf')
		return maxs

	def std(self):
		for name,d in self.distributions.iteritems():
			assert hasattr(d,'std'), "Not all component distributions have defined standard deviations"
		return dict([(name,d.std()) for name,d in self.distributions.iteritems()])
	def mean(self):
		for name,d in self.distributions.iteritems():
			assert hasattr(d,'mean'), "Not all component distributions have defined means"
		return dict([(name,d.mean()) for name,d in self.distributions.iteritems()]) 

	def fix(self,fixedParameters):
		for key,value in fixedParameters.iteritems():
			# NOTE: Rather than assertion, should informatively fail here.
			assert key in self.distributions.keys()
			self.distributions[key] = Point(value)


class DiscreteParameterDistribution(ParameterDistribution):
	''' Handles (potentially) dependent parameter distributions, such as those generated
		by posterior estimating parameterization techniques.
			particles: list<{name,float}>
			weights: list<float>
	'''
	def __init__(self,particles,weights=None):
		# Ensure that all particles are identically structured dicts
		err_msg = "DiscreteParameterDistribution: invalid instantiation"
		for p in particles:
			assert isinstance(p,dict), err_msg
			assert len(set(p.keys())-set(particles[0].keys()))==0, err_msg
		self.particles = particles

		# Normalize weights (may require float casting for ints)
		if weights != None:
			assert len(weights) == len(particles), err_msg
			self.weights = numpy.asarray([float(w) for w in weights])
		else:
			self.weights = numpy.ones(len(particles))
		self.weights = self.weights/sum(self.weights)

	def __repr__(self):
		ranges = {}
		for key in self.particles[0]:
			vals = [p[key] for p in self.particles]
			ranges[key] = (min(vals),max(vals))
		return str(ranges)

	# Returns a particle with probability proportional to its weight.
	def draw(self):
		r,curr = rand.rand(), 0.0
		for i,w in enumerate(self.weights):
			curr = curr+w
			if curr >= r:
				return self.particles[i]

	# If argument matches a particle, returns the particle weight. Otherwise, returns 0.
	def pdf(self,values):
		for i,p in enumerate(self.particles):
			if cmp(values,p) == 0:
				return self.weights[i]
		return 0.0

	# Return representation of the distribution as particle,weight tuples
	def summary(self):
		return zip(self.particles,self.weights)

	def _marginals(self):
		marginals = {}

		for i,particle in enumerate(self.particles):
			for pname in particle:
				if pname not in marginals:
					marginals[pname] = [particle[pname]]
				else:
					marginals[pname].append(particle[pname])
		return marginals

	# Returns a dicitonary of mean values of marginal distributions for each parameter.
	def mean(self):
		marginals = self._marginals()
		means = dict([(pname,0.0) for pname in marginals])
		try:
			for pname in marginals:
				for i,val in enumerate(marginals[pname]):
					means[pname] += self.weights[i] * val
			return means
		except:
			raise ValueError("DiscreteParameterDistribution - parameter structure incompatible with mean")


	# Returns a dictionary of standard deviations of marginal distributions for each parameter.
	def std(self):
		marginals = self._marginals()
		means = self.mean()
		stds = dict([(pname,0.0) for pname in marginals])
		try:
			for pname in marginals:
				for i,val in enumerate(marginals[pname]):
					stds[pname] += self.weights[i] * (val-means[pname])**2
			for pname in stds:
				stds[pname] = stds[pname]**(0.5)
			return stds
		except:
			raise ValueError("DiscreteParameterDistribution - parameter structure incompatible with mean")


	# Returns upper and lower bounds of the fraction q of marginal distribution density
	# centered about the mean.
	def ci(self,q):
		assert q > 0 and q <= 1, "Invalid density fraction for confidence interval"
		pct_lo = (1-q)/2
		pct_hi = q+pct_lo

		# Create a dictionary of pname->[(value,weight), ...] sorted by value
		sortedMarginals = dict([(key,[]) for key in self.particles[0]])
		for i,p in enumerate(self.particles):
			for key,val in p.iteritems():
				sortedMarginals[key].append((val,self.weights[i]))
		for key in sortedMarginals:
			sortedMarginals[key] = sorted(sortedMarginals[key],key=lambda pair: pair[0])

		# Find marginal parameter values marking pct_lo and pct_hi of the sorted distribution
		bounds = {}
		for key,values in sortedMarginals.iteritems():
			cumulative = 0.
			lowVal, hiVal = None,None
			for valWt in values:
				cumulative += valWt[1]
				if cumulative >= pct_lo and lowVal == None:
					lowVal = valWt[0]
				if cumulative >= pct_hi and hiVal == None:
					hiVal = valWt[0]
			bounds[key] = (lowVal,hiVal)

		return bounds

class MVNParameterDistribution(ParameterDistribution):
	
	def __init__(self,pnames,means=None,covMat=None):
		err_msg = "MVN: names, means, covariance must have consistent dimension"
		
		self.pnames = pnames
		if means != None:
			self.means = numpy.array(means)
			assert len(self.means) == len(self.pnames), err_msg
		else:
			self.means = numpy.zeros(len(pnames))
		if covMat != None:
			self.covMat = numpy.array(covMat)
			assert len(numpy.shape(self.covMat)) == 2, err_msg
			assert numpy.shape(self.covMat)[0] == len(self.means), err_msg
			assert numpy.shape(self.covMat)[1] == len(self.means), err_msg
		else:
			self.covMat = numpy.identity(len(pnames))

	def mean(self):
		return self.means
	def cov(self):
		return self.covMat

	def draw(self):
		result = stats.multivariate_normal.rvs(self.means,self.covMat)
		# n-d result
		if len(self.pnames) > 1:
			return dict([(self.pnames[i],val) for i,val in enumerate(result)])
		# 1-d result
		else:
			return {self.pnames[0]:result}

	def pdf(self,theta):
		vector = [theta[p] for p in self.pnames]
		#print self.covMat
		return stats.multivariate_normal.pdf(vector,self.means,self.covMat)

