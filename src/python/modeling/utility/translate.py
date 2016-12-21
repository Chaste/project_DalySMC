import numpy
from modeling.language.distributions import DiscreteParameterDistribution, IndependentParameterDistribution

''' Created to translate between dictionary representation of parameters used by the
	framework and the vector representation used by scipy.optimize methods.
	Accepts:
	- single floats/integers
	- regular nd-arrays thereof
'''
class DictToVector(object):
	def __init__(self,pdict):
		self.keys = pdict.keys()
		self.shapes = []

		for key in self.keys:
			try:
				shape = numpy.array(pdict[key]).shape
			except:
				 print "Could not convert a parameter element to a regular array"
			self.shapes.append(shape)

	def toVector(self,pdict):
		vectorRep = []
		for key in self.keys:
			if not hasattr(pdict[key],'__iter__'):
				vectorRep.append(pdict[key])
			else:
				for el in numpy.array(pdict[key]).flatten():
					vectorRep.append(el)
		return vectorRep

	def toDict(self,pvector):
		dictRep = {}
		vindex = 0
		for i,key in enumerate(self.keys):
			if sum(self.shapes[i]) == 0:
				dictRep[key] = pvector[i]
				vindex += 1
			else:
				els = pvector[vindex:vindex+numpy.prod(self.shapes[i])]
				dictRep[key] = numpy.reshape(numpy.array(els),self.shapes[i])
				vindex += numpy.prod(self.shapes[i])
		return dictRep


''' Replace the names of either a parameter set, DiscreteParameterDistribution, or
	IndependentParameterDistribution object according to string-string mapping specified
	in dict argument.

	NOTE: Will not play nice with constraints on IndependentParameterDistribution
'''
def Rename(target,mapping):
	if isinstance(target,dict):
		return _renameDict(target,mapping)
	elif isinstance(target,DiscreteParameterDistribution):
		newParticles = [_renameDict(p,mapping) for p in target.particles]
		return DiscreteParameterDistribution(newParticles,target.weights)
	elif isinstance(target,IndependentParameterDistribution):
		newDists = _renameDict(target.distributions)
		return IndependentParameterDistribution(newDists)
	else:
		raise TypeError("Rename() only operates on parameter sets and Discrete/Independent PDs")

def _renameDict(target,mapping):
	newDict = {}
	for key in target:
		if key in mapping:
			newDict[mapping[key]] = target[key]
	return newDict
