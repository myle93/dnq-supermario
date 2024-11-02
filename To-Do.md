seperate gamma in the optimizer (momentum) and gamma in the target (weigth of future actions)
--> parameter dokumentieren, was macht was und warum

add better visual validation of the model

save epsilon with model if its not finished training so you can resume training later

make validation a for loop (at least 10 episodes)

crop as hyperparameter? (and an easy way to customize it, including initial picture size)
also change network size based on it

visualize model architecture (draw.io oder so)

make visualization function better:
how to know if the performance is good and its training well (and fast)
and what you should do about it if it doesnt
--> als video?

cool graphs and save them:
tensorboard and stuff
log average q (from next State max average)
whatever proves useful

change hyperparameters:
optimizer (shampoo?)
loss function

change model architecture

performance:
torch.compile()
change 32 to 16 bit?
amp and https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
parralelize (use torch multiprocessing)
ray?
or other accelerators (DeepSpeed, Lightning etc.)
Mojo?

change game:
make own (efficient) enviroment?

automatic hyperparameter tuning (ray?)

parrallel / distributed training (ray?)

change algorithm:
double dqn
rainbow
Efficient Zero 2

get SOTA performance

improve it
in a specific game or generally

mit Robotern probieren

Weltherrschaft

