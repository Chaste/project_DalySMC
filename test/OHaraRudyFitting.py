# Unit testing imports
try:
    import unittest2 as unittest
except ImportError:
    import unittest

# Parameter fitting framework imports
import modeling.language.distributions as Dist
from modeling.language.kernels import Kernel
from modeling.simulation.fcexperiment import FunctionalCurationExperiment
from modeling.fitting.objective import SquareError
from modeling.fitting.algorithm import ParameterFittingTask
from modeling.fitting.approximatebayes import ABCSMCAdaptive, ABCSMCDelMoral
from modeling.utility import plotting, io	
from fc.language import values as V

import numpy
import matplotlib.pyplot as plt
import os, sys, time

import dill
import inspect, traceback

trueValues = {}
trueValues['oxmeta:membrane_fast_sodium_current_conductance'] = 75.0 # I_Na
trueValues['aidan:membrane_fast_sodium_late_current_conductance'] = 0.0075 # I_NaL
trueValues['oxmeta:membrane_transient_outward_current_conductance'] = 0.02 # I_to
trueValues['oxmeta:membrane_L_type_calcium_current_conductance'] = 0.0001 # I_CaL, I_CaNa, I_CaK
trueValues['oxmeta:membrane_rapid_delayed_rectifier_potassium_current_conductance'] = 0.046 # I_Kr
trueValues['oxmeta:membrane_slow_delayed_rectifier_potassium_current_conductance'] = 0.0034 # I_Ks
trueValues['oxmeta:membrane_inward_rectifier_potassium_current_conductance'] = 0.1908 # I_K1
trueValues['oxmeta:membrane_sodium_calcium_exchanger_current_conductance'] = 0.0008 # I_NaCai, I_NaCass
trueValues['oxmeta:membrane_sodium_potassium_pump_current_permeability'] = 30.0 # I_NaK
trueValues['aidan:membrane_background_sodium_current_conductance'] = 3.75e-10 # I_Nab
trueValues['oxmeta:membrane_background_potassium_current_conductance'] = 0.003 # I_Kb
trueValues['aidan:sarcolemmal_calcium_pump_current_conductance'] = 0.0005 # I_pCa
trueValues['aidan:membrane_background_calcium_current_conductance'] = 2.5e-8 # I_Cab

# Human-readable names for plotting
shortNames = {}
shortNames['oxmeta:membrane_fast_sodium_current_conductance'] = 'G_Na'
shortNames['aidan:membrane_fast_sodium_late_current_conductance'] = 'G_NaL'
shortNames['oxmeta:membrane_transient_outward_current_conductance'] = 'G_to'
shortNames['oxmeta:membrane_L_type_calcium_current_conductance'] = 'G_CaL'
shortNames['oxmeta:membrane_rapid_delayed_rectifier_potassium_current_conductance'] = 'G_Kr'
shortNames['oxmeta:membrane_slow_delayed_rectifier_potassium_current_conductance'] = 'G_Ks'
shortNames['oxmeta:membrane_inward_rectifier_potassium_current_conductance'] = 'G_K1'
shortNames['oxmeta:membrane_sodium_calcium_exchanger_current_conductance'] = 'G_NaCa'
shortNames['oxmeta:membrane_sodium_potassium_pump_current_permeability'] = 'G_NaK'
shortNames['aidan:membrane_background_sodium_current_conductance'] = 'G_Nab'
shortNames['oxmeta:membrane_background_potassium_current_conductance'] = 'G_Kb'
shortNames['aidan:sarcolemmal_calcium_pump_current_conductance'] = 'G_pCa'
shortNames['aidan:membrane_background_calcium_current_conductance'] = 'G_Cab'
shortNames['oxmeta:membrane_stimulus_current_amplitude'] = 'G_stim'

''' 
	Directory where all output files will be stored
'''
OUTPUT_DIR = "."

class OHaraRudyFitting(unittest.TestCase):

	def _OHaraFitting(self,objFun,algorithm,algName,protoFile,algArgs={},outputs={},
		paramsToFit=None,outputDir='.',data=None):
	
		# Allows for restriction of parameters to be fit
		if paramsToFit == None:
			paramsToFit = trueValues.keys()

		# Specification of the priors, proposal distributions 
		#  for the maximum conductances for the 14 ionic currents included in the model
		priors = dict([(p,Dist.Uniform(trueValues[p]/2,2*trueValues[p])) for p in paramsToFit])
		kernels = dict([(p,Dist.Normal(0,trueValues[p]*0.15)) for p in paramsToFit])
		prior = Dist.IndependentParameterDistribution(priors)
		kernel = Kernel(Dist.IndependentParameterDistribution(kernels))
		algArgs['proposalDist'] = kernel

		# Generate functional curation experiment
		modelFile = "projects/DalySMC/ohara_rudy_2011.cellml"
		experiment = FunctionalCurationExperiment(protoFile,modelFile)

		# Time the length of a simulation
		t_start = time.time()
		experiment.simulate()
		print "SIMULATION TIME:",protoFile,time.time()-t_start

		if outputs == {}:
			print "No outputs specified - defaulting to voltage"
			outputs = ['V']
		mapping = dict([(o,o) for o in outputs])

		# Generate simulated data
		if "sumstats" in protoFile:
			print "Using summary statistic data"
			simData = io.ReadDataSet("projects/DalySMC/test/data/OHrSumStats-8-2.dat")
			# Use experimental values as characteristic values for scaling
			objArgs = {'std':numpy.array(simData['sumStats'])}
		else:
			print "Using full trace data"
			sigma_e = 0.25 # The (known) magnitude of obervational error
			simData = io.ReadDataSet("projects/DalySMC/test/data/OHrAPtrace-8-2.dat")
			# Arguments to the objective function (magnitude of error, for likelihood)
			objArgs = {'std':sigma_e}

		prefix = os.path.join(outputDir,"OHrDM")
		if "sumstats" in protoFile:
			prefix += "SS"

		# Apply the fitting algorithm
		task = ParameterFittingTask(prior,experiment,simData,objFun,outputMapping=mapping,
			objArgs=objArgs)
		# Set ABC to terminate when it reaches the error under reported parameters
		if "sumstats" in protoFile:
			algArgs['minErr'] = task.calculateObjective(trueValues)
			print "ABC will terminate at: ", algArgs['minErr']
		# Record and report execution time
		t_start = time.time()
		results = algorithm(task,args=algArgs)
		print "EXECUTION TIME:",algName,time.time()-t_start
		#results = io.ReadParameterDistribution(algArgs['outputFile'])

		# Skip plotting if algorithm is an optimizer
		if len(results.particles) == 1:
			return results

		# Arbitrary partitioning for marginal biplots due to large number of currents
		currents1 = ['G_Na','G_NaL','G_NaCa','G_NaK','G_Nab']
		currents2 = ['G_Kr','G_Ks','G_K1','G_to','G_NaK','G_Kb']
		currents3 = ['G_CaL','G_NaCa','G_pCa','G_Cab']

		majorCurrents = ['G_Na','G_Kr','G_Ks','G_to','G_CaL']

		particles2 = [dict([(shortNames[key],value) for key,value in p.iteritems()]) for p in results.particles]
		results2 = Dist.DiscreteParameterDistribution(particles2,results.weights)
		trueValuesShort = dict([(shortNames[key],value) for key,value in trueValues.iteritems()])

		plotting.PlotDiscreteMarginals(results2,twoDim=True,trueValue=trueValuesShort,
			dest=prefix+"Na",restrict=currents1,nbins=50)
		plotting.PlotDiscreteMarginals(results2,twoDim=True,trueValue=trueValuesShort,
			dest=prefix+"K",restrict=currents2,nbins=50)
		plotting.PlotDiscreteMarginals(results2,twoDim=True,trueValue=trueValuesShort,
			dest=prefix+"Ca",restrict=currents3,nbins=50)
		plotting.PlotDiscreteMarginals(results2,twoDim=True,trueValue=trueValuesShort,
			dest=prefix+"Major",restrict=majorCurrents,nbins=50)

		plotting.PlotDiscreteMarginals(results2,twoDim=False,trueValue=trueValuesShort,
			dest=prefix,nbins=50)

		return results, task
	
	def TestDelMoralProb_13(self):
		for i in range(1):
			OUTPUT_FILE = os.path.join(OUTPUT_DIR,'OHrDM_Full'+str(i)+'.out')
			LOG_FILE = os.path.join(OUTPUT_DIR,'OHrDM_Full'+str(i)+'.log')
			TRACE_FILE = os.path.join(OUTPUT_DIR,'OHrDM_Full_postMeanTrace'+str(i)+'.eps')
			RESTRICT = None

			self.RunDelMoralProb(OUTPUT_DIR,OUTPUT_FILE,LOG_FILE,TRACE_FILE,RESTRICT)

	def TestDelMoralProb_5(self):
		for i in range(1):
			OUTPUT_FILE = os.path.join(OUTPUT_DIR,'OHrDM_Top5'+str(i)+'.out')
			LOG_FILE = os.path.join(OUTPUT_DIR,'OHrDM_Top5'+str(i)+'.log')
			TRACE_FILE = os.path.join(OUTPUT_DIR,'OHrDM_Top5_postMeanTrace'+str(i)+'.eps')
			RESTRICT = ['oxmeta:membrane_fast_sodium_current_conductance',
				'oxmeta:membrane_rapid_delayed_rectifier_potassium_current_conductance',
				'aidan:sarcolemmal_calcium_pump_current_conductance',
				'oxmeta:membrane_transient_outward_current_conductance',
				'aidan:membrane_background_calcium_current_conductance']

			self.RunDelMoralProb(OUTPUT_DIR,OUTPUT_FILE,LOG_FILE,TRACE_FILE,RESTRICT)

	def RunDelMoralProb(self,OUTPUT_DIR,OUTPUT_FILE,LOG_FILE,TRACE_FILE,RESTRICT):
		objFun = SquareError()
		outputs = ['V']
		protoFile = 'projects/DalySMC/test/protocols/oh_aptrace.txt'
		algorithm = ABCSMCDelMoral()

		def acceptFun(task,params,thresh):
			paramCopy = params.copy()
			if 'obj:std' in params:
				paramCopy['obj:std'] = paramCopy['obj:std']*thresh
			else:
				paramCopy['obj:std'] = 1.0*thresh
			return numpy.exp(-task.calculateObjective(paramCopy)/2)

		algArgs = {'cutoff':0.001,'postSize':1000,'e0':1000,'acceptFun':acceptFun,'tune':True,
			'minErr':1.0,'alpha':0.2,'outputFile':OUTPUT_FILE,
			'minESS':0.3,'resampleESS':0.6}

		post, task = self._OHaraFitting(objFun,algorithm,'DelMoralProb',protoFile,algArgs,
			outputs,outputDir=OUTPUT_DIR,paramsToFit=RESTRICT)

		sys.stdout = open(LOG_FILE,'w+')
		print "True values:"
		for pname in trueValues:
			print pname+"\t"+str(trueValues[pname])
		print "Mean posterior values:"
		postMean = post.mean()
		for pname in postMean:
			print pname+"\t"+str(postMean[pname])

		postMin = min([(p,task.calculateObjective(p)) for p in post.particles],key=lambda x:x[1])

		print "True SE: "+str(task.calculateObjective(trueValues))
		print "Mean SE: "+str(task.calculateObjective(postMean))
		print "Min SE: "+str(postMin[1])

		# Variance reduction ratio for each parameter
		sigma_prior = task.prior.std()
		sigma_post = post.std()
		for lname in sigma_prior:
			print "\t".join([shortNames[lname],str(sigma_post[lname]/sigma_prior[lname])])

		# Plot overlaid traces of model under true parameters and mean posterior predicted parameters
		fig = plt.figure()
		trueTrace = task.experiment.simulate(trueValues)
		postTrace = task.experiment.simulate(postMean)
		plt.plot(trueTrace['t'],task.expData['V'],c='r',label='Noisy data')
		plt.plot(trueTrace['t'],trueTrace['V'],c='b',label='True parameters')
		plt.plot(trueTrace['t'],postTrace['V'],c='g',label='Mean posterior parameters')
		plt.legend(fontsize=10)
		plt.xlabel("Time (ms)",fontsize=12)
		plt.ylabel("Membrane voltage (mV)",fontsize=12)
		plt.tick_params(axis='both', which='major', labelsize=8)
		plt.savefig(TRACE_FILE,format='eps',dpi=300)

	def TestDelMoral01_13(self):
		for i in range(1):
			OUTPUT_FILE = os.path.join(OUTPUT_DIR,'OHrDM_SS_Full'+str(i)+'.out')
			LOG_FILE = os.path.join(OUTPUT_DIR,'OHrDM_SS_Full'+str(i)+'.log')
			RESTRICT = None

			self.RunDelMoral01(OUTPUT_DIR,OUTPUT_FILE,LOG_FILE,RESTRICT)

	def TestDelMoral01_5(self):
		for i in range(1):
			OUTPUT_FILE = os.path.join(OUTPUT_DIR,'OHrDM_SS_Top5'+str(i)+'.out')
			LOG_FILE = os.path.join(OUTPUT_DIR,'OHrDM_SS_Top5'+str(i)+'.log')
			RESTRICT = ['oxmeta:membrane_fast_sodium_current_conductance',
				'oxmeta:membrane_rapid_delayed_rectifier_potassium_current_conductance',
				'aidan:sarcolemmal_calcium_pump_current_conductance',
				'oxmeta:membrane_transient_outward_current_conductance',
				'aidan:membrane_background_calcium_current_conductance']

			self.RunDelMoral01(OUTPUT_DIR,OUTPUT_FILE,LOG_FILE,RESTRICT)

	def RunDelMoral01(self,OUTPUT_DIR,OUTPUT_FILE,LOG_FILE,RESTRICT):
		objFun = SquareError()
		outputs = ['sumStats']
		protoFile = 'projects/DalySMC/test/protocols/oh_aptrace_sumstats.txt'
		algorithm = ABCSMCDelMoral()

		algArgs = {'cutoff':0.00001,'postSize':1000,'tune':True,
			'alpha':0.2,'outputFile':OUTPUT_FILE,
			'minESS':0.3,'resampleESS':0.6,
			'e0':1}

		post, task = self._OHaraFitting(objFun,algorithm,'DelMoral01',protoFile,algArgs,
			outputs,outputDir=OUTPUT_DIR,paramsToFit=RESTRICT)

		old_stdout = sys.stdout
		sys.stdout = open(LOG_FILE,'w+')
		print "True values:"
		for pname in trueValues:
			print pname+"\t"+str(trueValues[pname])
		print "Mean posterior values:"
		postMean = post.mean()
		for pname in postMean:
			print pname+"\t"+str(postMean[pname])

		postMean = post.mean()
		print "True SE: "+str(task.calculateObjective(trueValues))
		print "Mean SE: "+str(task.calculateObjective(postMean))
		#print "Min SE: "+str(min([task.calculateObjective(p) for p in post.particles]))

		# Variance reduction ratio for each parameter
		sigma_prior = task.prior.std()
		sigma_post = post.std()
		for lname in sigma_prior:
			print shortNames[lname] + "\t" +str(sigma_post[lname]/sigma_prior[lname])

		trueTrace = task.experiment.simulate(trueValues)
		postTrace = task.experiment.simulate(postMean)
		print trueTrace['sumStats']
		print postTrace['sumStats']

		sys.stdout = old_stdout

		# Prevents too many figures from being opened (gives a memory warning)
		# TODO: Should probably close figures in plotting module
		plt.close("all")

