import numpy
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

import numpy as np
from scipy.stats import norm
from scipy.misc import comb

''' Plots difference between simulated and experimental data over dependent axis
	(should be uncorrelated under correct model, noise assumption).
	NOTE: ASSUMES 1D OUTPUTS
	- 'distribution': ParameterDistribution
	- 'experiment': Experiment
	- 'data': dictionary of ndarrays consistent with 'experiment' output
	- 'outputs': (optional) list of experimental outputs to plot. Defaults to all.
'''
def Plot1DResiduals(distribution,experiment,data,outputs=None,ndraws=10,dest='residuals'):
	if outputs == None:
		outputs = data.keys()

	for output in outputs:
		fig = plt.figure()
		xs = range(len(data[output]))

		for i in range(ndraws):
			particle = distribution.draw()
			simData = experiment.simulate(particle)
			plt.plot(xs,simData[output]-data[output])

		plt.ylabel("Residual "+output)
	fname = dest+"_"+output+".eps"
	plt.savefig(fname,format='eps',dpi=1000)


''' Plots empirical quantiles against quantiles of a theoretical distribution.
	Points are expected to fall on a line of slope 1 if the distributions match.
	- 'outputs': 1d iterable containing the emprical distribution 
	- 'ppf': inverse CDF of a (univariate) distribution to be compared against
	- 'nquantiles': number of quantiles. Defaults to number of particles in 'distribution.'
'''
def PlotQQ(outputs,weights=None,ppf=norm.ppf,nquantiles=None,dest='qqplot',shift=False):
	# Check if one or multiple distributions are input
	if not hasattr(outputs[0],'__iter__'):
		outputs = [outputs]
		weights = [weights]

	# Scatter ordered empirical values against corresponsing inverse CDF values
	fig = plt.figure()

	for ind,outputs1d in enumerate(outputs):
		weights1d = weights[ind]
		noutputs = len(outputs1d)

		if nquantiles == None:
			nquantiles = noutputs

		# Default to uniform weighting
		if weights1d == None:
			weights1d = numpy.ones(noutputs)/noutputs
		else:
			assert len(weights1d)==noutputs, "PlotQQ: Outputs and Weights expected to have the same shape"
			assert abs(sum(weights1d)-1.0)<0.0000001, "PlotQQ: Weights expected to be normalized"

		# Sort the 1d outputs by value
		weightedOutputs = zip(outputs1d,weights1d)
		weightedOutputs.sort()

		current_quant = 1
		summed_weights = 0.0
		firstPlotted = False
		y1,yn = 0,0
		xvalues,yvalues = [],[]

		for i,empiricalVal in enumerate(weightedOutputs):
			summed_weights += empiricalVal[1]

			# To account for floating point error in weight sums
			if i==len(weightedOutputs)-1:
				summed_weights = 1.0

			if summed_weights < float(current_quant)/nquantiles:
				continue

			while summed_weights >= float(current_quant)/nquantiles:
				if not firstPlotted:
					y1 = ppf(float(current_quant)/nquantiles)
					firstPlotted = True
				# Many continuous distributions will have ppf(1)=inf
				if ppf(float(current_quant)/nquantiles) < float('inf'):
					yn = ppf(float(current_quant)/nquantiles)
					xvalues.append(empiricalVal[0])
					yvalues.append(ppf(float(current_quant)/nquantiles))
				current_quant += 1

		plt.scatter(xvalues,yvalues,marker='.',c='b')

	# Plot slope-1 line for reference
	plt.plot([y1,yn],[y1,yn],'r')
	
	# TODO: should calculate some box around both the line of fit and the scattered points
	xmin,xmax,ymin,ymax = plt.axis()
	plt.axis((y1-(yn-y1)/2,yn+(yn-y1)/2,y1-(yn-y1)/2,yn+(yn-y1)/2))

	# Set behavior for x-axis shift
	ax = plt.gca()
	ax.get_xaxis().get_major_formatter().set_useOffset(shift)
	ax.get_yaxis().get_major_formatter().set_useOffset(shift)

	plt.xlabel("Empirical Quantiles")
	plt.ylabel("Theoretical Quantiles")

	plt.tight_layout()

	plt.savefig(dest+'.eps',format='eps',dpi=1000)


''' Plots empirical quantiles of one distribution against empirical quantiles of another.
	Points are expected to fall on a line of slope 1 if the distributions match.
	- 'outputs[1,2]': 1d iterable containing an empirical distribution
	- 'weights[1,2]': 1d iterable containing corresponding weights (defaults to uniform)
	- 'nquantiles': number of quantiles (defaults to size of distribution with fewest particles)
'''
def PlotQQ2(outputs1,outputs2,weights1=None,weights2=None,nquantiles=None,dest='qqplot'):
	n1 = len(outputs1)
	n2 = len(outputs2)

	if nquantiles == None:
		nquantiles = min(n1,n2)

	# Default to uniform weighting
	if weights1 == None:
		weights1 = numpy.ones(n1)/n1
	else:
		assert len(weights1)==n1, "PlotQQ2: Outputs and weights expected to have the same shape"
		assert abs(sum(weights1)-1.0)<0.00000001, "PlotQQ2: Weights expected to be normalized"
	if weights2 == None:
		weights2 = numpy.ones(n2)/n2
	else:
		assert len(weights2)==n2, "PlotQQ2: Outputs and weights expected to have the same shape"
		assert abs(sum(weights2)-1.0)<0.00000001, "PlotQQ2: Weights expected to be normalized"

	# Sort the 1d outputs by value
	weightedOutputs1 = zip(outputs1,weights1)
	weightedOutputs1.sort()
	weightedOutputs2 = zip(outputs2,weights2)
	weightedOutputs2.sort()

	# Scatter ordered empirical values against value corresponding to an equivalent quantile
	# of the second distribution
	fig = plt.figure()

	xvalues, yvalues = [],[]
	current_quant = 1
	summed_weights = 0.0
	firstPlotted = 0
	x1,xn=0,0

	# Find quantiles for the first distribution
	for i,empiricalVal in enumerate(weightedOutputs1):
		summed_weights += empiricalVal[1]

		# To account for floating point error in weight sums
		if i==len(weightedOutputs1)-1:
			summed_weights = 1.0

		if summed_weights < float(current_quant)/nquantiles:
			continue

		while summed_weights >= float(current_quant)/nquantiles:
			if not firstPlotted:
				x1 = empiricalVal[0]
				firstPlotted = True
			xn = empiricalVal[0]
			xvalues.append(empiricalVal[0])
			current_quant += 1

	current_quant = 1
	summed_weights = 0.0

	# Find quantiles for the second distribution
	for i,empiricalVal in enumerate(weightedOutputs2):
		summed_weights += empiricalVal[1]

		# To account for floating point error in weight sums
		if i==len(weightedOutputs2)-1:
			summed_weights = 1.0

		if summed_weights < float(current_quant)/nquantiles:
			continue

		while summed_weights >= float(current_quant)/nquantiles:
			yvalues.append(empiricalVal[0])
			current_quant += 1

	plt.scatter(xvalues,yvalues,marker='x')

	# Plot a slope-1 line for reference
	plt.plot([min(outputs2),max(outputs2)],[min(outputs2),max(outputs2)],'-r')

	# TODO: should calculate some box around both the line of fit and the scattered points
	#xmin,xmax,ymin,ymax = plt.axis()
	#plt.axis((x1-(xn-x1)/4,xn+(xn-x1)/4,x1-(xn-x1)/4,xn+(xn-x1)/4))

	plt.xlabel("Empirical Distribution 1")
	plt.ylabel("Empirical Distribution 2")

	plt.savefig(dest+'.eps',format='eps',dpi=1000)

''' Accepts a DiscreteParameterDistribution object and plots a heat map
	of pairwise marginal likelihoods
	- If a subset of parameters is specified in "include", only pairs 
	  from within that subset will be plotted
	- "bins" specifies the resolution
	- "plotRange" - either a dict with (min,max) tuples for each axis, or a single
	   such tuple for all axes
	- "separateFigs" - specify whether to plot all figures as sublots in a horizontal
	   figure, or to save each as a separate figure names as "basename_P1_P2.eps"
'''
def PlotHeatMap(distribution,dest='heatmap',include=None,bins=50,plotRange=None,
	separateFigs=True):
	if include == None:
		include = distribution.draw().keys()

	marginals = distribution._marginals()

	matplotlib.rcParams.update({'font.size': 14})

	# For setting limits of the shared color bar in the single figure plot
	cmin, cmax, ims = 1,0,[]

	if not separateFigs:
		nplots = int(comb(len(include),2))
		fig = plt.figure(figsize=(6.5*nplots,6))
		gs = gridspec.GridSpec(1,nplots)
		ind = 0

	for i in range(len(include)):
		for j in range(i+1,len(include)):			
			px,py = include[i],include[j]

			if isinstance(plotRange,dict) and px in plotRange and py in plotRange:
				xyRange = [plotRange[px],plotRange[py]]
			else:
				xyRange = plotRange

			heatmap, xedges, yedges = np.histogram2d(marginals[px], marginals[py], 
				bins=bins, weights=distribution.weights,range=xyRange)
			extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

			if separateFigs:
				fig, ax = plt.subplots(figsize=(6,6))
			else:
				ax = plt.subplot(gs[0,ind])
				if numpy.min(heatmap) < cmin:
					cmin = numpy.min(heatmap)
				if numpy.max(heatmap) > cmax:
					cmax = numpy.max(heatmap)
				ind += 1

			im = ax.imshow(heatmap.T, interpolation="nearest", extent=extent, origin="lower")
			ax.set_aspect(float(xedges[-1]-xedges[0])/(yedges[-1]-yedges[0]))
			
			ax.set_xlabel(px)
			ax.set_ylabel(py)

			if separateFigs:
				fig.colorbar(im)
				plt.savefig(dest+"_"+px+"_"+py+".eps",format="eps")
			else:
				ims.append(im)

	if not separateFigs:
		for im in ims:
			im.set_clim((cmin,cmax))
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(ims[-1], cax=cbar_ax)
		plt.savefig(dest+".eps",format="eps")

	return None


''' Accepts a DiscreteParameterDistribution object and plots either:
	- Histograms of its 1d marginal distributions (twoDim='False')
	- Biplots of its 2d marginal distributions (twoDim='True')
	- 'restrict': specified subset of parameters to plot; defaults to all
'''
def PlotDiscreteMarginals(distribution,dest='marginal',trueValue=None,errorBars=None,
	twoDim=False,restrict=None,nbins=10,shift=False):
	if restrict == None:
		restrict = distribution.particles[0].keys()

	# Parse the marginal distributions from the DiscreteParameterDistribution object
	marginals = {}
	densities = []
	for i,particle in enumerate(distribution.particles):
		# Control for zero-weighted particles
		if distribution.weights[i] > 0:
			for key,val in particle.iteritems():
				if key in restrict:
					if not key in marginals:
						marginals[key] = [val]
					else:
						marginals[key].append(val)
			densities.append(distribution.weights[i])

	if not twoDim:
		_plotHistos(marginals,densities,dest,trueValue,errorBars,nbins,shift)
	else:
		_plotBiplots(marginals,densities,dest,trueValue,errorBars,nbins,shift)

def _plotHistos(marginals,densities,dest,trueValue=None,errorBars=None,nbins=10,shift=False):
	for parameter,values in marginals.iteritems():
		
		fig = plt.figure()
		matplotlib.rcParams.update({'font.size': 20})

		# Scale each particle proportionally to its weight
		plt.hist(values,weights=densities,bins=nbins)

		if trueValue != None:
			plt.axvline(x=trueValue[parameter],c='r')
			if errorBars != None:
				plt.axvline(x=trueValue[parameter]-errorBars[parameter],c='k')
				plt.axvline(x=trueValue[parameter]+errorBars[parameter],c='k')

		plt.xlabel(parameter)
		plt.ylabel("Frequency")

		# Adjust x axis limits to make sure trueValue bar is visible on plot
		if trueValue != None:
			x1,x2,y1,y2 = plt.axis()
			if trueValue[parameter] >= x2:
				x2 = trueValue[parameter] + (trueValue[parameter]-x1)*0.1
			if trueValue[parameter] <= x1:
				x1 = trueValue[parameter] - (x2-trueValue[parameter])*0.1
			plt.axis((x1,x2,y1,y2))
			print parameter,str(x1),str(x2),trueValue[parameter]

		# Set behavior for x-axis shift
		ax = plt.gca()
		ax.get_xaxis().get_major_formatter().set_useOffset(shift)

		fname = dest+"_"+parameter+".eps"
		locs, labels = plt.xticks()
		plt.setp(labels, rotation=30)
		plt.tight_layout()
		plt.savefig(fname,format='eps',dpi=1000)

def _plotBiplots(marginals,densities,dest,trueValue=None,errorBars=None,nbins=10,shift=False):
	nparams = len(marginals)

	#fig,axarr = plt.subplots(nparams,nparams)
	fig = plt.figure()
	gs = gridspec.GridSpec(nparams,nparams)

	font = {'size':6}
	matplotlib.rc('font', **font)

	# Marginal histograms (diagonal)
	# Must be plotted before any biplots for axis sharing reasons
	for ind,p in enumerate(marginals.keys()):
		values = marginals[p]

		ax = plt.subplot(gs[ind,ind])
		plt.setp(ax.get_yticklabels(),visible=False)
		plt.setp(ax.get_xticklabels(),rotation=90)
		ax.set_xlabel(p,labelpad=10,fontsize=10)

		ax.hist(values,weights=densities,bins=nbins)

		if trueValue != None:
			ax.axvline(trueValue[p],c='r')
			if errorBars != None:
				plt.axvline(x=trueValue[p]-errorBars[p],c='k')
				plt.axvline(x=trueValue[p]+errorBars[p],c='k')

	# Correlation scatter (off-diagonal)
	for xind,p1 in enumerate(marginals.keys()):
		values1 = marginals[p1]

		for yind,p2 in enumerate(marginals.keys()):
			if xind > yind:
				continue

			if xind != yind:
				# Share x axis with histogram below
				histax = plt.subplot(gs[yind,yind])
				ax = plt.subplot(gs[xind,yind],sharex=histax)

				values2 = marginals[p2]
				ax.scatter(values2,values1,c=densities)
				if trueValue != None:
					ax.scatter(trueValue[p2],trueValue[p1],c='r')
				
				# Top-most plot (x-axis only, on top)
				if xind == 0:
					ax.xaxis.tick_top()
					ax.set_xlabel(p2,labelpad=10,fontsize=10)
					ax.xaxis.set_label_position('top')
				plt.setp(ax.get_xticklabels(),visible=False)
				
				# Left-most plot (y-axis only)
				if yind == 0:
					ax.set_ylabel(p1)
				
				# Right-most plot (y-axis only, on right)
				if yind == nparams-1:
					ax.yaxis.tick_right()
					ax.set_ylabel(p1,fontsize=10)
					ax.yaxis.set_label_position('right')
				plt.setp(ax.get_yticklabels(),visible=False)
			
				# Without correction, y-axis is set too small when scale of parameters differs
				minv,maxv = min(values1),max(values1)
				ax.set_ylim([minv-0.25*(maxv-minv),maxv+0.25*(maxv-minv)])


	fname = dest+".eps"
	plt.savefig(fname,format='eps',dpi=1000)


''' Plot Gelman-Rubin distance for a series of Markov chains stored in a
	set of CSV files.
	Used to assess convergence over time.
'''
def PlotGelmanRubin(fileNames,dest='gelman_rubin'):
	
	# Parse CSV files into a dict of 2d arrays
	allchains = {}
	for fh in fileNames:
		thischain = {}
		csvfile = open(fh)
		reader = csv.reader(csvfile,delimiter='\t')
		header = reader.next()

		for pname in header:
			thischain[pname] = []

		for row in reader:
			for i,entry in enumerate(row):
				thischain[header[i]].append(float(entry))

		for key,val in thischain.iteritems():
			if not key in allchains:
				allchains[key] = [thischain[key]]
			else:
				allchains[key].append(thischain[key])

	GR_traces = {}

	# For each parameter in the posterior:
	for pname,vals2d in allchains.iteritems():
		GR_traces[pname] = []
		nchains, nsamples = numpy.shape(vals2d)
		print nchains,nsamples

		print pname

		# At each point in time, across all traces:
		for timeInd in range(2,nsamples):

			# Calculate the average variance within each chain
			inner_vars = [numpy.var(vals2d[chainInd][0:timeInd],ddof=1) for chainInd in range(nchains)]
			W = numpy.mean(inner_vars)

			# Calculate the variance across all chains (between mean values up to timeInd)
			inner_means = [numpy.mean(vals2d[chainInd][0:timeInd]) for chainInd in range(nchains)]
			B = numpy.var(inner_means,ddof=1) * timeInd

			v = W*(timeInd-1)/timeInd + B/timeInd

			GR = numpy.sqrt(v/W)
			GR_traces[pname].append(GR)

	# Plot the GR over time
	fig, axarr = plt.subplots(1,len(GR_traces))
	for ind,key in enumerate(GR_traces.keys()):
		# Plot the GR trace and reference of GR=1
		# Leave out the first 10 iterations (after the first) to avoid 0-variance chains
		axarr[ind].plot(range(10,nsamples-2),GR_traces[key][10:])
		axarr[ind].plot(range(10,nsamples-2),numpy.ones(nsamples-12),'-r')

		ymin,ymax = axarr[ind].get_ylim()
		axarr[ind].set_ylim([0,ymax])

		axarr[ind].set_title(key)
		axarr[ind].set_xlabel('Last iteration in chain')
		axarr[ind].tick_params(axis='x',labelsize=6)
		plt.setp(axarr[ind].get_xticklabels(),rotation='vertical')
	axarr[0].set_ylabel('Gelman-Rubin')

	fname = dest+'.eps'
	plt.savefig(fname,format='eps',dpi=1000)

''' Plot change in marginal distributions across a series of experiments
	- 'distributions': iterable of DiscreteParameterDistribution
	- 'expVarVals': (optional) iterable containing the variable affecting the posterior
	  (must be same length as 'distributions')
	- 'expVarName': (optional) name of the experimental variable
'''
def PlotMarginalEvolution(distributions,expVarVals=None,expVarName='Experiment Index',
	dest='marginalEvolution'):
	if expVarVals != None:
		assert len(expVarVals) == len(distributions)
	else:
		expVarVals = np.arange(len(distributions))
	
	# For each parameter in the posterior, create a plot of 'marginal vs. expVar'
	for pname in distributions[0].particles[0].keys():

		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')

		# Parse marginal distribution into parallel lists of values, associated weights
		for ind, distribution in enumerate(distributions):
			#marginal = [p[pname] for p in distribution.particles]
			#densities = distribution.weights

			# Explicit resampling to make sure all have same number of particles
			draws = [distribution.draw() for i in range(1000)]
			marginal = [draw[pname] for draw in draws]

			hts,divs = np.histogram(marginal,bins=20,density=True)
			#hts,divs = np.histogram(marginal,weights=densities,bins=20,density=True)
			# np.histogram returns n+1 coordinates, marking start and ends of bins
			lcs = divs[0:len(hts)]

			ax.bar(lcs,hts,width=lcs[1]-lcs[0],zs=expVarVals[ind],zdir='y',alpha=0.8)

		ax.set_xlabel(pname)
		ax.set_ylabel(expVarName)
		ax.set_zlabel('Frequency')

		fname = dest+"_"+pname+".eps"
		plt.savefig(fname,format='eps',dpi=1000)


''' Plot transluscent histograms of distributions on the same axes to test agreement
	- 'distributions': iterable of DiscreteParameterDistribution
	- 'labels': iterable of distribution identifiers for the legend
	- 'separateFigs': True to plot each marginal in a separate file, False to combine
'''
def PlotMarginalsOverlaid(distributions,labels=None,separateFigs=True,trueValue=None,dest='marginalOverlay'):
	if labels == None:
		labels = ['Data Set '+str(i+1) for i in range(len(distributions))]

	# Assumes all distributions are identically structured
	pnames = distributions[0].draw().keys()

	if not separateFigs:
		fig = plt.figure(figsize=(10,3*len(pnames)))
		gs = gridspec.GridSpec(len(pnames)+1,1)

	# Create color specs
	# NOTE: As of now, repeats will appear after 18 colors. Need to apply the "/2, *1.5" pattern
	# to the base set of six colors, not the whole growing list
	colors = [(0.,0.,1.),(0.,1.,0.),(1.,0.,0.),(0.,1.,1.),(1.,0.,1.),(1.,1.,0.)]
	while len(distributions) > len(colors):
		colors2 = [(r/2,g/2,b/2) for (r,g,b) in colors]
		colors = colors + colors2
		if len(distributions) > len(colors):
			colors3 = [(r*1.5,g*1.5,b*1.5) for (r,g,b) in colors2]		
			colors = colors + colors3

	# Create a separate plot for each parameter
	for ind,p in enumerate(pnames):
		if separateFigs:
			fig = plt.figure(figsize=(10,3))
			plt.xlabel(p,labelpad=10,fontsize=10)
			plt.ylabel('Frequency')
			plt.xticks(rotation=30)
		else:
			ax = plt.subplot(gs[ind,0])
			ax.set_xlabel(p,labelpad=10,fontsize=10)
			ax.set_ylabel('Frequency')

		for i,dist in enumerate(distributions):
			values = [particle[p] for particle in dist.particles]
			weights = dist.weights

			if separateFigs:
				plt.hist(values,weights=weights,histtype='step',label=labels[i],bins=100,color=colors[i],normed=True)
				if trueValue != None:
					plt.axvline(trueValue[p],c='k')
			else:
				ax.hist(values,weights=weights,histtype='step',label=labels[i],bins=100,color=colors[i],normed=True)
				if trueValue != None:
					plt.axvline(trueValue[p],c='k')

		if separateFigs: 
			plt.legend(prop={'size':8})
			plt.gcf().subplots_adjust(bottom=0.25)
			plt.savefig(dest+'_'+p+'.eps',format='eps',dpi=400)
	
	if not separateFigs:
		gs.tight_layout(fig,h_pad=0.75)
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.55),ncol=5,fontsize=10)
		plt.savefig(dest+'.eps',format='eps',dpi=400)

