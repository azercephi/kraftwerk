# Tests a bunch of strategies using sim.py
# Usage python test_strategies.py strategy/2.10.21.json.seeds. 2.10.21.json


import sys
from os import listdir
from os.path import isfile, join
import os
from sim import simulate_strategies


def main():
	folder = sys.argv[1]
	graph = sys.argv[2]
	n = int(graph.split('.')[1])
	
	strategies = sorted([f for f in listdir(folder) if isfile(join(folder, f))], reverse=True)

	assert('top_deg' in strategies)
	
	print strategies
	
	winning = []
	
	for strategy in strategies:
		if strategy != 'top_deg':
			result = simulate_strategies(str(graph), folder+'/'+ \
				strategy, folder+'/top_deg')
				
			deg_cnt = result[folder+'/top_deg']
			strat_cnt = result[folder+'/'+strategy]
			if strat_cnt > deg_cnt:
				winning.append([folder+'/'+strategy, strat_cnt])
				print result
					
	winning.sort(key=lambda x: x[1], reverse=True)
	
	if len(winning) > 10: # Our winnings are bountiful
		winning = winning[0:10]
	winning  = [x[0] for x in winning]
	print 'winning', winning

	all_winning = []
	for strat in winning:
		print 'strat',strat
		with open(strat, 'r') as f:
			line = f.readline().strip()[1:-1]
			line = '\n'.join(line.split(','))
			print line
			all_winning.append(line)

	print all_winning
	with open('seeds/all_winning', 'w') as f:
		for i in range(50):
			f.write(all_winning[i % len(all_winning)])
			if i != 49:			
				f.write('\n')
	print 'winning', winning


main()
