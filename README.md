# pr-project
Pattern Recognition Project

In order to run the evaluation for each Scenario, there are a couple of steps you need to take. 
Note: The variable `X` will be referred several times through the steps. It is either `1` or `2`, depending on the scenario you run.
1. Open Matlab and access `train_classifier_X`
2. Run the script. After it is finished, it will add `wX`  to your workspace, which is the classifier used for evaluation
3. To evaluate, run `nist_eval(my_repX, wX, n)` with an `n` of choice.

For the live test, the steps are the following
1'. If you want to run the algorithm that extracts the digits from the scan, run `livewriting.m`. However, a command prompt will prompt you to annotate each digit in the command window, which might take a while. Otherwise, you will find the annotated dataset, named `live_dataset.mat`, which you will have to load for the next step.
1. Run `pipeline_live.m`. At the end, you will be presented with the classification error in the variable `errLive`.

If you want to go through the whole research pipeline we followed for the entire project, run `pipeline_scenarioX.m`.
