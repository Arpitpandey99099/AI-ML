import numpy as np
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# A* PATHFINDING IMPLEMENTATION
# -----------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    g_cost = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == end:
            return g_cost[current]

        x, y = current
        neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                new_cost = g_cost[current] + 1
                if (nx, ny) not in g_cost or new_cost < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_cost
                    f_cost = new_cost + heuristic((nx, ny), end)
                    heapq.heappush(open_list, (f_cost, (nx, ny)))

    return -1  # No path found


# -----------------------------
# MAZE GENERATION
# -----------------------------
def generate_maze(size=20, wall_prob=0.3):
    grid = np.random.choice([0, 1], size=(size, size), p=[1-wall_prob, wall_prob])
    grid[0][0] = 0
    grid[size-1][size-1] = 0
    return grid


# -----------------------------
# DATASET CREATION
# -----------------------------
def create_dataset(num_samples=500):
    X = []
    y = []

    for _ in range(num_samples):
        grid = generate_maze()
        length = astar(grid, (0, 0), (19, 19))

        if length != -1:  # only keep solvable mazes
            X.append(grid)
            y.append(length)

    X = np.array(X)
    y = np.array(y)

    return X, y


# -----------------------------
# CNN MODEL
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model(X, y, epochs=10):
    device = torch.device("cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()

        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


# -----------------------------
# TEST + VISUALIZATION
# -----------------------------
def test_model(model):
    grid = generate_maze()
    true_length = astar(grid, (0, 0), (19, 19))

    input_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pred_length = model(input_tensor).item()

    print(f"True Path Length: {true_length}")
    print(f"Predicted Length: {pred_length:.2f}")

    plt.imshow(grid, cmap='gray')
    plt.title("Maze")
    plt.show()


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("Generating dataset...")
    X, y = create_dataset(500)

    print("Training model...")
    model = train_model(X, y, epochs=10)

    print("Testing model...")
    test_model(model)
