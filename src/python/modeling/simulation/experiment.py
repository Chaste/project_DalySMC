'''
	Created as part of the modeling experiment description extension to functional 
	curation Chaste.
'''

# Python Abstract Base Class support
from abc import ABCMeta, abstractmethod

''' 
	ABSTRACT CLASSES:
		Experiment
'''
class Experiment:
	__metaclass__ = ABCMeta

	''' Returns dict of experimental outputs when simulated under supplied parameters. '''
	@abstractmethod
	def simulate(self,parameters): pass

	''' Accepts a dictionary of name,value pairs and updates experimental inputs for next run. ''' 
	''' Required only for experimental design analyses (not yet supported from Algorithm). '''
	def setInputs(self,inputs):
		raise NotImplementedError


