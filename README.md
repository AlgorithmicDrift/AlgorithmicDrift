1) Configure the run settings in main.py (models, etas and so on)
2) Run main.py

Configuration details:
1) module:
   1) training, to train models
   2) evaluation, to evaluate pre-trained models
   3) generation, to run the simulation
2) strategy: No_strategy/Organic
3) to_parallelize:
   1) eta, different etas will be assigned to each parallel process
   2) model, different models will be assigned to each parallel process
4) indices_call: you need to specify how many processes will be runned in parallel and which etas/models will be splitted among them in each call

Note: we refer to the different dataset proportions as "eta".
Example of eta: 0.65_0.2_0.15 -> 65% non-radicalized, 20% semi-radicalized, 15% radicalized
