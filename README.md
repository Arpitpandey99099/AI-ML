# Algorithmic Pathfinding Predictor

This is my Bring Your Own Project (BYOP) submission for the Fundamentals of AI and ML course.

## What is this?
Normally, finding the shortest path in a large grid or maze using algorithms like A* or Dijkstra's takes a lot of time and compute power. I wanted to see if we could skip the search algorithm entirely using Machine Learning.

Instead of calculating the path step-by-step, this project uses a Convolutional Neural Network (CNN). The CNN looks at a 2D array representation of a maze (where `0` is an open path and `1` is a wall) and predicts the total length of the shortest path almost instantly.

## How I built it
1. **Data Generation:** I wrote a Python script to create random 20x20 mazes. It uses the standard A* algorithm to find the actual shortest path length, which becomes the "target" label for the ML model.
2. **The Model:** I built a simple CNN using [PyTorch / TensorFlow]. Since the grids are relatively small (20x20), it trains pretty fast locally on my CPU without needing a massive GPU.
3. **Visuals:** I used `matplotlib` to plot the arrays so you can actually see the mazes and compare the model's prediction with the real path.

## Screenshots
*[Add a screenshot of your matplotlib output here showing the maze and the predicted vs actual path length]*

## How to run it on your machine

First, clone the repo and install the dependencies (mostly just numpy, matplotlib, and your ML framework).

```bash
git clone [https://github.com/yourusername/pathfinding-predictor.git](https://github.com/yourusername/pathfinding-predictor.git)
cd pathfinding-predictor
pip install -r requirements.txt
