import csv

import modeling.language.distributions as Dist

''' Reads a discrete parameter distribution from a file containing newline-separated
	delimited parameter vectors.
	- Parameter names specified in first row
	- Weights (optionally) specified by a column index
	- Supports headers commented by "#"
'''
def ReadParameterDistribution(fileName,delim='\t',weightCol=None):
	particles,weights = [],[]

	csvfile = open(fileName)
	reader = csv.reader(csvfile,delimiter=delim)

	pnames = reader.next()
	while pnames[0][0] == '#':
		pnames = reader.next()

	if "WEIGHTS" in pnames:
		weightCol = pnames.index("WEIGHTS")
	if weightCol == None:
		weights = None

	for line in reader:
		particle = {}
		for i,p in enumerate(pnames):
			if i != weightCol:
				particle[p] = float(line[i])
			else:
				weights.append(float(line[i]))
		particles.append(particle)

	csvfile.close()
	return Dist.DiscreteParameterDistribution(particles,weights)

''' Writes a DiscreteParameterDistribution to a (default) tab-delimited file
	- Parameter names specified in first row
	- Weights specified in final column under heading "WEIGHTS"
'''
def WriteParameterDistribution(fileName,distribution,header='',delim='\t'):
	csvfile = open(fileName,'w')

	# TODO: Make sure each new line of header is marked by "#"
	if header != '':
		tokens = header.split('\n')
		for t in tokens:
			csvfile.write('#')
			csvfile.write(t)
			csvfile.write('\n')

	pnames = distribution.draw().keys()
	csvfile.write(delim.join(pnames))
	csvfile.write('\tWEIGHTS')
	csvfile.write('\n')

	for ind,particle in enumerate(distribution.particles):
		values = [str(particle[p]) for p in pnames]
		values.append(str(distribution.weights[ind]))
		csvfile.write(delim.join(values))
		csvfile.write('\n')

	csvfile.close()

''' Read one or more 0- or 1-D numerical outputs into a dict
'''
def ReadDataSet(fileName,delim='\t'):
	csvfile = open(fileName)

	reader = csv.reader(csvfile,delimiter=delim)
	outputs = reader.next()

	while outputs[0][0] == "#":
		outputs = reader.next()

	dataSet = dict([(key,[]) for key in outputs])

	for row in reader:
		for i,entry in enumerate(row):
			dataSet[outputs[i]].append(float(entry))

	return dataSet


'''	Write one or more 0- or 1-D outputs to a file
	- Length of all outputs should be consistent
	- 'outputs' may potentially specify a subset of named outputs to write
'''
def WriteDataSet(fileName,dataSet,outputs=None,header='',delim='\t'):
	if outputs == None:
		outputs = dataSet.keys()

	csvfile = open(fileName,"w+")

	# Write (optional) header
	if header != '':
		tokens = header.split('\n')
		for t in tokens:
			csvfile.write('#')
			csvfile.write(t)
			csvfile.write('\n')

	noutputs = len(outputs)
	nentries = len(dataSet[outputs[0]])

	# Write specifier of output names
	csvfile.write(delim.join(outputs))
	csvfile.write("\n")

	for j in range(nentries):
		row = [str(dataSet[outputs[i]][j]) for i in range(noutputs)]
		csvfile.write(delim.join(row))
		csvfile.write("\n")

	csvfile.close()


