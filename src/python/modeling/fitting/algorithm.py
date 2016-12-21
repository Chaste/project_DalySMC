'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

# Python Abstract Base Class support
from abc import ABCMeta, abstractmethod

import objective as Objective

''' 
	HELPER CLASS
		ParameterFittingTask
'''

class ParameterFittingTask(object):
	''' Wrapper class for the following essential parameter fitting arguments:
			- ParameterDistribution prior
			- Experiment experiment
			- dict expData
			- Objective objFun
		And the following optional arguments:
			- dict outputMapping: maps names of simulated output to keys in expData
			- dict inputs: maps names of protocol inputs to desired values
			- dict objArgs: arguments to the objective function

		Used as simplified input for ParameterFittingAlgorithm class.
	'''
	def __init__(self,prior,experiment,expData,objFun,outputMapping=None,inputs=None,objArgs={}):
		self.prior = prior
		self.experiment = experiment
		self.expData = expData
		self.outputMapping = outputMapping
		self.objFun = objFun
		self.objArgs = objArgs

		# Sets protocol inputs. This is done only once.
		if inputs != None:
			for key,value in inputs.iteritems():
				self.experiment.setInputs(inputs)

	''' Handles interaction between components to produce Objective output from
		parameter values.
		Primary method utilized by ParameterFittingAlgorithm.
	'''
	def calculateObjective(self,parameters):
		data1, data2 = {}, {}

		# If parameters are specified with the reserved namespace 'obj',
		# pass them to the objective function.
		# If objective args are specified with the same name, they will be overwritten
		# on calls to calculateObjective.
		simParams = {}

		for key,val in parameters.iteritems():
			tokens = key.split(':')
			if len(tokens)>1 and tokens[0]=='obj':
				if self.objArgs == None:
					self.objArgs = {}
				self.objArgs[tokens[1]] = val
			else:
				simParams[key] = val

		try:
			simData = self.experiment.simulate(simParams)
		except:
			# Catch all solver failures by returning a default objective value.
			# If not overridden by user, +inf for distance and -inf for likelihood/proximity.
			if hasattr(self.objFun,'defaultValue'):
				return self.objFun.defaultValue
			elif self.objFun.isMinimized:
				return float('inf')
			else:
				return -float('inf')

		# Match experimental/simulated data for input to objective function
		if self.outputMapping != None:
			for simName,expName in self.outputMapping.iteritems():
				data1[simName] = simData[simName]
				data2[simName] = self.expData[expName]
		else:
			data1 = simData
			data2 = self.expData

		# Special case: reqularized objectives require passing of parameter values
		# NOTE: Should probably alter RegularizedObjective to take as an extra argument
		if isinstance(self.objFun,Objective.RegularizedObjective):
			if self.objArgs == None:
				self.objArgs = {}
			self.objArgs['modelParameters'] = simParams 

		return self.objFun(data1,data2,self.objArgs)

'''
	ABSTRACT CLASS
		ParameterFittingAlgorithm
'''

class ParameterFittingAlgorithm:
	__metaclass__ = ABCMeta

	''' Accepts a ParameterFittingTask object and algorithm-specific arguments.
		Should be implemented as a static/class method (all state maintained by 
			ParameterFittingTask, which should be passed to helper methods)
	'''
	@abstractmethod
	def __call__(task,args=None): pass


class ExperimentDesignAlgorithm:
	__metaclass__ = ABCMeta

	@abstractmethod
	def __call__(task,args=None): pass

