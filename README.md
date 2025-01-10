# Imitation Learning
This repository contains the imitation learning source code and trained model for F1 Tenth autononomus racing <br><br>

# Architectural Overview

### Expert 
- `gap_follow.py`: gap_follow node for expert data
- `opp_gap_follow.py`: opponent gap_follow node for 1v1 race 

### Data Collection
- `data_collector.py`: node to collect expert data 
- `data_filter.py`: node to collect expert data with data filter

### Neural Network
- `imitation_nn.py`: defines the multilayer perceptron (MLP) of the IL model

### Training
- `train.py`: trains the IL model

### Evaluation
- `imitation_node.py`: runs the trained IL model
- `performance.py`: computes the accuracy of the IL Model compared to its expert (`gap_follow.py`) in redbull_ring_obs map

### `mlp_constants.py`
Since there are a number of common constants shared across multiple nodes, we created a separate file to assign values to them, avoiding the need to manually update values in each node

- `DEVICE`: device used to train or load the model
    - Our IL model was trained with Mac MPS. If you are using Windows and have an Nvidia GPU, please change the device to CUDA
- MLP constants: used to define neural network class object
- `EPOCHS`: specifies the number of iterations over the entire training dataset
- `DATASET`: names the dataset used for collection or training
- `MODEL`: names the model to train or the model to run <br><br>


# How to collect expert data and train 
- The trained IL model is already in the `saved model` folder. However, if you wish to collect expert data and train the model yourself, please follow the instructions below.
    - We would like to note that since certain hyperparameters are influenced by the size of the dataset, your own trained model might not perform as good as the provided model from this repo due to the differences in dataset size. 
- To run `data_collector.py` or `data_filter.py`:
    - In `mlp_constants.py`, set `DATASET_TO_COLLECT` to name your dataset
    - In both files, change the path to your own
    - Run `data_collector.py` or `data_filter.py` first and run expert node (which is gap_follow for our project) afterward
    - After you collected enough data, __please turn off data_collector.py or data_filter.py node first__ before turning off the expert node. Turning off the expert node first will cause the car to crash, and this will be included in the training data

-  To train the car, choose the size of neural network layers, `epochs` size, and specify `DATASET_TO_TRAIN` and `MODEL_TO_TRAIN` in `mlp_constants.py`. Then, run `train.py` to train 
    - In `train.py`, if you want to load and train the existing model, uncomment the load model statement on line 67. Otherwise, leave it commented <br><br>



# How to run IL model 
- Set `MODEL_TO_RUN` in `mlp_constants.py` to choose the model to run
    - It is already set to our IL model but if you trained model on your own and want to test it, change it to your own model
- Start `imitation_node.py` to run the IL model
- When running and evaluating the IL model, please __make sure you are only running the simulator and the imitation node__ not other programs or processes (e.g. training other models, running other nodes that are not used in the simulator)
    - We identified that running `imitation_node.py` while training different model or running a lot of programs in parallel caused a significant latency, which resulted in the car publishing wrong behavior or crashing in the worst case
    - To properly evaluate a model, we always tested after the training process is done or after stopping other nodes not used in the evaluation 
    - If you really need to run other processes together, you may need more computational resources (e.g. allow more memory to Docker)<br><br>

# How to run the expert
- To evaluate the expert performance:
    - Run `gap_follow.py` on redbull_ring_obs or levine_blocked
    - To run it with the IL model, you need to run `opp_gap_follow.py` not `gap_follow.py`
        - Please follow the levine_blocked 1v1 race instruction below <br><br>

# Maps used in evaluation
- Redbull Ring Obstacles Map:
    - We used this map to test how accurate the behavioral cloning of our IL model performed
    - If you want to run the IL model in this map, please download `redbull_ring_obs.png` and `redbull_ring_obs.yaml` from this repository and save them under `maps` folder of your simulator pkg
    - To ensure the car drives counter-clockwise, please set ego staring pose's `stheta` to 3.14159 in the `sim.yaml` file of your simulator pkg
        - The initial position is clockwise, so we set it to 3.14159 to turn the car 180 degrees

- Levine Blocked Map:
    - We used this map to compare the path planning of the IL model and the expert. We also used this map to do racing with an opponent car to test if it can overtake the car ahead
    - If you changed the `stheta` value to 3.14159 to use redbull ring map earlier, please change it back to 0.0 so that it can drive counter-clockwise in this map
    - 1v1 race with an opponent car:
        - Increase the number of cars in `sim.yaml` to 2 and set the opponent stating pose to be `sx1 = 4.0` and `sy1 = -0.5` to place the opponent's statring grid 4m ahead of IL model's starting grid
            - You can choose your own coordinates for the opponent car's starting grid, but the farther it starts, the longer it will take for the IL model to overtake it
        - In the RViz simulator, add the Robot Model and set description topic of the opponent robot model to `opp_robot_description`
        - Run `opp_gap_follow.py` to run the opponent car
        - Run `imitation_node.py` to run the IL model





