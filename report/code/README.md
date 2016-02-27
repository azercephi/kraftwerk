# kraftwerk
CS 144 Pandemaniac

To generate strategies in 'strategy' folder for use in simulation, call
clear; python seeder.py 2.10.10.json

To simulate a run, call
python sim.py strategy1 strategy [generatePics]
where [generatePics] will generate pictures in 'figs' folder if True.
For example, we can call:

To test several strategies, call
python test_strategies.py foldername graphname
where foldername contains all the graphs and graphname is a valid json graph file.


