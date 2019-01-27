# pr-project
Pattern Recognition Project

In order to run the evaluation for each Scenario, there are a couple of steps you need to take. 
Note: The variable `X `will be referred several times through the steps. It is either `1` or `2`, depending on the scenario you run.
1. Open Matlab and access `train_classifier_X`/
2. Run the script. After it is finished, it will add `wX`  to your workspace, which is the classifier used for evaluation
3. To evaluate, run `nist_eval(my_repX, wX, n)` with an `n` of choice.