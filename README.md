# Algorithmic Pathfinding Predictor

This is my Bring Your Own Project (BYOP) submission for the Fundamentals of AI and ML course.

## The Problem
Running optimal pathfinding algorithms like A* or Dijkstra's on very large, complex grid mazes takes a significant amount of computational time. I wanted to see if we could skip the step-by-step search process entirely by using Machine Learning.

## My Approach
Instead of calculating the path, this project uses a Convolutional Neural Network (CNN). The model takes a 2D array representation of a maze (where `0` is an open space and `1` is a wall) and directly predicts the total length of the shortest path from start to finish.

It treats the pathfinding problem as a Computer Vision and Regression task. 

## Project Structure
* `generate_data.py`: Script to create random 20x20 mazes and use the standard A* algorithm to calculate the true shortest path (which acts as our target label).
* `train.py`: The script that builds and trains the CNN model on the generated maze data.
* `predict.py`: Takes a brand new, unseen maze and compares the CNN's predicted path length with the actual A* path length.

## How to Set Up and Run

You don't need a heavy GPU for this. It runs perfectly fine on a local CPU.

**1. Clone the repository**
```bash
git clone [https://github.com/Arpitpandey99099/algorithmic-pathfinding-predictor.git](https://github.com/Arpitpandey99099/algorithmic-pathfinding-predictor.git)
cd algorithmic-pathfinding-predictor
