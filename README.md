# kraftwerk
CS 144 Pandemaniac

To generate strategies in 'strategy' folder for use in simulation, call
clear; python seeder.py 2.10.10.json

To simulate a run, call
python sim.py strategy1 strategy [generatePics]
where [generatePics] will generate pictures in 'figs' folder if True.
For example, we can call:
clear; python sim.py 2.10.json strategy/2.10.10.json.seeds.top_beatdeg strategy/2.10.10.json.seeds.unweighted_top_deg True


