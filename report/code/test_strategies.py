# Tests a bunch of strategies using sim.py
# Usage python test_strategies.py strategy/2.10.21.json.seeds. 2.10.21.json

import sys, os
from os import listdir
from os.path import isfile, join
from sim import simulate_strategies
ROUNDS = 50

def main():
	usage = 'python test_strategies.py foldername graphname'
	if len(sys.argv) < 2:
		print >> stderr, usage
		exit(1) 
	folder = sys.argv[1]
	graph = sys.argv[2]
	n = int(graph.split('.')[1])
	
	strategies = sorted([f for f in listdir(folder) if isfile(join(folder, f))], reverse=True)
	assert('more_deg' in strategies)
	
	winning = []
	
	for strategy in strategies:
		if strategy.find('more') == -1:
			result = simulate_strategies(str(graph), folder+'/'+ \
				strategy, folder+'/more_deg')
				
			result2 = simulate_strategies(str(graph), folder+'/'+ \
				strategy, folder+'/more_close')
				
			deg_cnt = result[folder+'/more_deg']
			strat_cnt = result[folder+'/'+strategy]
			
			close_cnt = result2[folder+'/more_close']
			strat_cnt2 = result2[folder+'/'+strategy]
			#if strat_cnt2 > close_cnt or strat_cnt > deg_cnt:
			winning.append([folder+'/'+strategy, strat_cnt+strat_cnt2])
				#print result
					
	winning.sort(key=lambda x: x[1], reverse=True)
	
	if len(winning) > 10: # Our winnings are bountiful
		winning = winning[0:10]
	print 'winning', winning

	winning  = [x[0] for x in winning]

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
		for i in range(ROUNDS):
			f.write(all_winning[i % len(all_winning)])
			if i != ROUNDS-1:			
				f.write('\n')
	print 'winning', winning

main()
