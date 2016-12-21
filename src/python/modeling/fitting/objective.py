'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

# Python Abstract Base Class support
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy

'''
	TODO: Class property distinguishing between likelihood/distance functions
	Algorithms could make automated decisions about maximizing/minimizing.
'''

'''
	ABSTRACT CLASSES:
		Objective
		MinimizedObjective
		MaximizedObjective
		WeightedObjective
'''

''' Base class for all Objective functions. 
	SHOULD NOT BE EXTENDED DIRECTLY - inherit from MaximizedObjective for likelihood functions
	  and MinimizedObjective for distance functions, or manually override isMinimized
'''
class Objective(object):
	__metaclass__ = ABCMeta

	''' Accepts two dictionaries of data in <name, value> format and returns a float.
		We assume the two data sets are of identical format before reaching the Objective.
	'''
	@abstractmethod
	def __call__(self,data1,data2,args={}): pass

	''' Boolean declaring whether the objective is to be minimized (True; distance) or
		maximized (False; likelihood)
	'''
	@abstractproperty
	def isMinimized(self): pass

''' Base classes for distance and likelihood functions.
	Inheriting from these classes allows algorithms to optimize in the correct direction.
		- defaultValue: optional 'failure' value that may be overridden. Defaults to +/i inf
'''
class MinimizedObjective(Objective):
	@property
	def isMinimized(self): return True

	@property
	def defaultValue(self): return float('inf')

class MaximizedObjective(Objective):
	@property
	def isMinimized(self): return False

	@property
	def defaultValue(self): return -float('inf')

''' Weighted sum of an objective metric evaluated over (multiple) pairs of nd-arrays (including 0d) '''
class WeightedObjective(Objective):

	''' Objective metric between any given simulated/experimental data (nd-arrays) '''
	@abstractmethod
	def objectiveMetric(self,array1,array2,args={}): pass

	''' Accepts two nd-arrays or two dictionaries with the same <name,nd-array> structure
		Accepts a 'weighting' argument which may be valued as:
			- A dictionary of <outputname, weight> format
			- A callable object operating on an nd-array (data2)
			- A string describing a common weighting scheme: 'mean'|'meanSquare'|'sdev'
		Unless separate arguments for the objectiveMetric are passed as 'objectiveMetricArgs',
		the arguments to __call__ will be forwarded.
	'''
	@classmethod
	def __call__(cls,data1,data2,args={}):
		# If the data are both nd-arrays, skip the weighting step
		if isinstance(data1,list) or isinstance(data1,numpy.ndarray):
			assert isinstance(data2,list) or isinstance(data2,numpy.ndarray), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof"

			omargs = args
			if 'objectiveMetricArgs' in args:
				omargs = args['objectiveMetricArgs']
			return cls.objectiveMetric(numpy.array(data1),numpy.array(data2),omargs)
		
		# Otherwise, data expected in named dictionary format
		else:
			assert isinstance(data1,dict) and isinstance(data2,dict), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof" 
		obj = 0.0
		for key in data1.keys():
			assert key in data2.keys(), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof"
			assert numpy.shape(data1[key]) == numpy.shape(data2[key]), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof"

			# Defaults to uniform weighting in the absence of specification
			wt = 1.0
			omargs = args.copy()

			# Allows passing of objective function argumetns that differentially apply to 
			#  different parameters
			for argname,argval in args.iteritems():
				if isinstance(argval,dict):
					if key in argval:
						omargs[argname] = argval[key]
			#print "Arguments for output "+key+": "+str(omargs)

			if 'weighting' in args:
				# Dictionary of custom weights
				# NOTE: should we allow for dictionary of callable?
				if isinstance(args['weighting'],dict):
					assert key in args['weighting'].keys(), "Weighting scheme incompletely specified"
					assert isinstance(args['weighting'][key],float), "Invalid weighting specified for output"
					wt = args['weighting'][key]
				# Function of experimental data
				elif hasattr(args['weighting'],'__call__'):
					wt = args['weighting'](data2[key])
				# Predefined weighting scheme
				elif args['weighting'] == 'mean':
					wt = 1.0/pow(numpy.mean(data2[key]),2)
				elif args['weighting'] == 'meanSquare':
					wt = 1.0/numpy.mean(numpy.square(data2[key]))
				elif args['weighting'] == 'sdev':
					wt = 1.0/numpy.std(data2[key])
				else:
					assert 0, "Invalid formatting of weighting scheme for Objective"
			if 'objectiveMetricArgs' in args:
				omargs = args['objectiveMetricArgs']

			# Uses computed/provided weight to calculate total error
			obj = obj + wt * cls.objectiveMetric(data1[key],data2[key],omargs)
		return obj


''' 
	COMMON OBJECTIVE IMPLEMENTATIONS (POINT-POINT):
		SquareError
		RMSE
		AbsoluteError
		LogLikGauss
'''
class SquareError(WeightedObjective,MinimizedObjective):
	@staticmethod
	def objectiveMetric(array1,array2,args={}):
		std = 1.0
		if 'std' in args:
			std = args['std']
		return numpy.sum((array1-array2)**2/(std**2))

class MSE(SquareError):
	@staticmethod
	def objectiveMetric(array1,array2,args={}):
		df = 0
		if 'degreesFreedom' in args:
			df = args['degreesFreedom']
		nelements = numpy.array(array1).size
		return super(MSE,MSE).objectiveMetric(array1,array2,args)/(nelements-df)

class RMSE(MSE):
	@staticmethod
	def objectiveMetric(array1,array2,args=None):
		return numpy.sqrt(super(RMSE,RMSE).objectiveMetric(array1,array2,args))

class AbsoluteError(WeightedObjective,MinimizedObjective):
	@staticmethod
	def objectiveMetric(array1,array2,args=None):
		return numpy.sum(numpy.abs(array1-array2))

''' Two reserved parameter names:
	- 'mean' - mean of the Gaussian
	- 'std' - standard deviation of the Gaussian
'''
class LogLikGauss(WeightedObjective,MaximizedObjective):
	@staticmethod
	def objectiveMetric(array1,array2,args={}):
		if not isinstance(array1,list) or not isinstance(array1,numpy.ndarray):
			array1 = numpy.array([array1])
		if not isinstance(array2,list) or not isinstance(array2,numpy.ndarray):
			array2 = numpy.array([array2])
		
		mean, std = 0.0, 1.0
		if 'mean' in args:
			mean = args['mean']
		if 'std' in args:
			std = args['std']
		if isinstance(std,numpy.ndarray) or isinstance(std,list):
			assert len(std) == len(array1)
			reg = -len(array1)*numpy.log(numpy.sqrt(2*numpy.pi)) + numpy.sum(std)
		else:
			reg = -len(array1)*numpy.log(numpy.sqrt(2*numpy.pi)*std)
		lik = -numpy.sum((array1-array2-mean)**2/(2*std**2))
		return reg+lik

''' Two reserved parameter names:
	- 'mean' - mean of the uniform
	- 'width' - width of the uniform about the mean 
'''
class LogLikUniform(WeightedObjective,MaximizedObjective):
	@staticmethod
	def objectiveMetric(array1,array2,args={}):
		mean, width = 0.0, 2.0
		if 'mean' in args:
			mean = args['mean']
		if 'width' in args:
			width = args['width']
		hi,lo = mean+width/2, mean-width/2
		for x in array1-array2:
			if x < lo or x > hi:
				return 0
		return len(array1)*numpy.log(1.0/(hi-lo))


class NegLogLikGauss(LogLikGauss):
	@staticmethod
	def objectiveMetric(array1,array2,args={}):
		return -LogLikGauss.objectiveMetric(array1,array2,args)

	@property
	def isMinimized(self): return True


'''
	COMMON OBJECTIVE FUNCTIONS (POINT-DISTRIBUTION)
		MeanEuclidean
		Mahalanobis (TODO)
'''

class MeanEuclidean(MaximizedObjective):
	''' Compares a single realization of each data against repeated observations
		- 'singleData' - single simulated trace (dict)
		- 'repeatData' - repeatedly observed data (dict of iterables)
		TODO: Automatically detect which input is repeated, rather than detecting by
		  argument order. Symmetric inputs would allow this to be used for comparing
		  repeated (stochastic) simulations against a single data set.
		Reserved parameter names:
		- 'std' - scale for each quantity (dict) or all quantities (float)
	'''
	@classmethod
	def __call__(cls,singleData,repeatData,args={}):
		assert isinstance(singleData,dict) and isinstance(repeatData,dict), "MeanEuclidean expected dict-valued inputs"

		outputs = singleData.keys()
		# Parse arguments detailing spread over each output
		if 'std' in args:
			if isinstance(args['std'],dict):
				std = args['std']
			else:
				assert isinstance(args['std'],float) or isinstance(args['std'],int), "MeanEuclidean expects std to be specified as dict or number"
				std = dict([(name,args['std']) for name in outputs])
		else:
			std = dict([(name,1.0) for name in outputs])

		# Define shape of repeated data and assert all data is of that form
		noutput = len(outputs)
		nrepeats = len(repeatData[outputs[0]])
		for key,data in repeatData.iteritems():
			assert len(data) == nrepeats, "MeanEuclidean detected missing data"

		total = 0.
		for j in range(nrepeats):
			inner_total = 0.
			
			for i,name in enumerate(outputs):	
				# If each data are ndarray, the euclidean distance will be summed over all dimensions
				euc = ((numpy.array(singleData[name])-numpy.array(repeatData[name][j]))/std[name])**2
				inner_total += numpy.sum(euc)

			total += numpy.sqrt(inner_total)
		return total/nrepeats


'''
	REGULARIZED OBJECTIVE IMPLEMENTATIONS:
		RegularizedObjective
		L1RegularizedObjective
		L2RegularizedObjective
'''

''' Regularized objective functions balance an Objective function value on data with
	a penalty function on the model (parameters).
	Used for sparsity-induced model selection. 
	
	One reserved parameter name:
	- 'tradeoff' - scale of the penalty term
'''
class RegularizedObjective(Objective):
	
	''' Accepts an Objective and a penalty function callable on a parameter dictionary '''
	def __init__(self,objectiveFunction,penaltyFunction):
		self.objectiveFunction = objectiveFunction
		self.penaltyFunction = penaltyFunction

	''' Minimize or maximize according to base function '''
	@property
	def isMinimized(self): return self.objectiveFunction.isMinimized()

	''' Requires 'modelParameters' (parameter dictionary) in args for input the penalty function. 
		Optional 'tradeoff' in args serves as a float-value weight on the penalty.
	'''
	# NOTE: should RegularizedObjective and its objectiveFunction accept the same arguments?
	def __call__(self,data1,data2,args={}):
		tradeoff = 1.0
		assert 'modelParameters' in args.keys(), "Regularized objective function requires 'modelParameters' argument"
		if 'tradeoff' in args.keys():
			assert isinstance(args['tradeoff'],float) or isinstance(args['tradeoff'],int), "Regularized objective function expects a floating point tradeoff argument"
			tradeoff = args['tradeoff']
		return self.objectiveFunction(data1,data2,args) + tradeoff * self.penaltyFunction(args['modelParameters'])


''' Regularized objective penalizing number of non-zero parameters '''
class L1RegularizedObjective(RegularizedObjective):
	
	''' Accepts an Objective and a list of parameters to penalize '''
	def __init__(self,objectiveFunction,regulatedParameters=None):
		self.regulatedParameters = regulatedParameters
		if regulatedParameters != None:
			penaltyFunction = lambda params: sum([params[x]>0 for x in self.regulatedParameters])
		else:
			penaltyFunction = lambda params: sum([params[x]>0 for x in params.keys()])
		super(L1RegularizedObjective,self).__init__(objectiveFunction,penaltyFunction)


''' Regularized objective penalizing squared sum of parameter values '''
class L2RegularizedObjective(RegularizedObjective):
	
	''' Accepts an Objective and a list of parameters to penalize '''
	def __init__(self,objectiveFunction,regulatedParameters=None):
		self.regulatedParameters = regulatedParameters
		if regulatedParameters != None:
			penaltyFunction = lambda params: sum([params[x]**2 for x in self.regulatedParameters])
		else:
			penaltyFunction = lambda params: sum([params[x]**2 for x in params.keys()])
		super(L2RegularizedObjective,self).__init__(objectiveFunction,penaltyFunction)


'''
	COMPOSITVE OBJECTIVE IMPLEMENTATIONS:
		CompositeObjective
'''

''' Shorthand for combining objective functions '''
class CompositeObjective(Objective):

	''' Objectives can be supplied in ORDERED (list/array) or UNORDERED (dict) format
	    weighting can be supplied as either:
	    - Iterable of same type/structure as Objectives containing float-valued weights
	    - Callable accepting an iterable of the same type/structure as Objectives, returning
	      a single float resulting from combination of Objective outputs.
	'''
	def __init__(self,objectives,weighting=None):
		self.objectives = objectives
		self.weighting = weighting

	''' NOTE: assumes all component objectives are optimized in the same direction '''
	@property
	def isMinimized(self):
		if isinstance(self.objectives,dict):
			return self.objectives[self.objectives.keys()[0]].isMinimized
		else:
			return self.objectives[0].isMinimized 

	''' In order to allow for different data/arguments to be supplied to different Objectives,
	    data and args may separately take the following forms:
	    - Single dict to be passed to all Objectives
	    - Dict of dicts to be passed to Objectives with corresponding keys (UNORDERED)
	    - Iterable of dicts to be passed to Objectives with corresponding indices (ORDERED)
		The distributeData flag tells the function whether to pass the entire data (False) or
		index it to a parallel Objective (True)
	'''
	def __call__(self,data1,data2,args={},distributeData=False):
		objdata1 = data1
		objdata2 = data2
		objargs = args

		# When Objectives supplied as a dictionary:
		if isinstance(self.objectives,dict):
			objvals = {}
			for key,objfun in self.objectives.iteritems():
				# Data must be either single dictionaries or nd-arrays to be passed to all
				#   objectives, or dicts of dicts to be passed to corresponding objectives
				if distributeData:
					assert isinstance(data1,dict) and isinstance(data2,dict)
					if key in data1.keys() and key in data2.keys():
						objdata1 = data1[key]
						objdata2 = data2[key]

				# Check if separate args are supplied for each objective (dict of dicts)
				if key in args.keys():
					objargs = args[key]
				objvals[key] = objfun(objdata1,objdata2,objargs)

			# When no weighting specified, defaults to summation of objectives
			if self.weighting == None:
				return sum([val for key,val in objvals.iteritems()])
			elif hasattr(self.weighting,'__call__'):
				return self.weighting(objvals)
			else:
				return sum([self.weighting[key]*val for key,val in objvals.iteritems()])

		# When Objectives supplied as an ordered iterable:
		else:
			objvals = []
			for ind,objfun in enumerate(self.objectives):
				# Data must be either single dictionaries of nd-arrays to be passed to all
				#   objectives, or lists/arrays of dicts to be passed to corresponding objectives
				if distributeData:
					assert (isinstance(data1,list) or isinstance(data1,numpy.ndarray)) 
					assert (isinstance(data2,list) or isinstance(data2,numpy.ndarray))
					objdata1 = data1[ind]
				   	objdata2 = data2[ind]

				# Check if separate args are supplied for each objective (list of dicts)
				if isinstance(args,list):
					objargs = args[ind]
				objvals.append(objfun(objdata1,objdata2,objargs))

			# When no weighting specified, defaults to summation of objectives
			if self.weighting == None:
				return sum(objvals)
			elif hasattr(self.weighting,'__call__'):
				return self.weighting(objvals)
			else:
				return sum([val*self.weighting[ind] for ind,val in enumerate(objvals)])


