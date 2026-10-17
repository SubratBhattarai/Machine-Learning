Iris Boundary Animator

An interactive, educational Python project that implements Logistic Regression from scratch to classify Iris setosa vs. Iris versicolor using two features (Sepal Length & Sepal Width). The project records the decision boundary state at every training epoch and animates its convergence over time.

📋 Table of Contents

Overview

Mathematical Foundation

Human Numerical Example

Project Structure

Installation & Requirements

How to Run

Code Explanation

🌟 Overview

Machine learning models learn iteratively through Gradient Descent. Instead of simply jumping to the final answer, the decision boundary starts at a random state and slowly tilts, shifts, and aligns itself to separate class clusters.

This project visualizes that exact learning process frame-by-frame:

Dataset: Iris Dataset (Filtered to Setosa vs. Versicolor)

Features Used: Sepal Length ($x_1$) and Sepal Width ($x_2$)

Algorithm: Binary Logistic Regression (NumPy only, built from scratch)

Visualization: Matplotlib Animation (FuncAnimation)

📐 Mathematical Foundation

1. Model Equation

The probability $\hat{y}$ that a flower belongs to class 1 (Versicolor) is calculated using a linear combination passed through the Sigmoid Activation Function:

$$z = w_1 \cdot x_1 + w_2 \cdot x_2 + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:

$w_1, w_2$: Feature weights

$b$: Bias offset

$x_1, x_2$: Normalized Sepal Length and Sepal Width

2. Decision Boundary Formula

The decision threshold occurs where the model predicts equal probability ($\hat{y} = 0.5$), which corresponds to $z = 0$:

$$w_1 \cdot x_1 + w_2 \cdot x_2 + b = 0$$

To plot this 2D line on standard axes where $x_1$ is horizontal ($x$) and $x_2$ is vertical ($y$), we solve for $x_2$:

$$x_2 = -\frac{w_1}{w_2} \cdot x_1 - \frac{b}{w_2}$$

Slope ($m$): $-\frac{w_1}{w_2}$

Intercept ($c$): $-\frac{b}{w_2}$

3. Gradient Descent Updates

At each epoch, weights and bias are updated using Binary Cross-Entropy loss gradients:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum (\hat{\mathbf{y}} - \mathbf{y})$$

$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \mathbf{w}}$$

$$b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial L}{\partial b}$$

(where $\eta$ is the learning rate).

🧮 Human Numerical Example

To see how parameters update in a single iteration, let's step through a toy example with 1 sample:

Initial Setup

Sample Feature ($x_1, x_2$): $(2.0, 1.0)$

True Label ($y$): $1$ (Versicolor)

Weights ($w_1, w_2$): $0.5, -0.5$

Bias ($b$): $0.1$

Learning Rate ($\eta$): $0.1$

Step 1: Forward Pass

$$z = (0.5 \times 2.0) + (-0.5 \times 1.0) + 0.1 = 1.0 - 0.5 + 0.1 = 0.6$$

$$\hat{y} = \frac{1}{1 + e^{-0.6}} \approx \frac{1}{1 + 0.5488} \approx 0.6456$$

Step 2: Calculate Error

$$\text{Error} = \hat{y} - y = 0.6456 - 1.0 = -0.3544$$

Step 3: Calculate Gradients

$$\frac{\partial L}{\partial w_1} = \text{Error} \cdot x_1 = -0.3544 \times 2.0 = -0.7088$$

$$\frac{\partial L}{\partial w_2} = \text{Error} \cdot x_2 = -0.3544 \times 1.0 = -0.3544$$

$$\frac{\partial L}{\partial b} = \text{Error} = -0.3544$$

Step 4: Parameter Updates

$$w_1^{\text{new}} = 0.5 - (0.1 \times -0.7088) = 0.57088$$

$$w_2^{\text{new}} = -0.5 - (0.1 \times -0.3544) = -0.46456$$

$$b^{\text{new}} = 0.1 - (0.1 \times -0.3544) = 0.13544$$

In the next epoch, these new parameters ($w_1^{\text{new}}, w_2^{\text{new}}, b^{\text{new}}$) become the input for another round of predictions!

📂 Project Structure

.
├── main.py          # Complete script: Data preparation, model training, and animation
└── README.md        # Documentation and mathematical guide


💻 Installation & Requirements

Make sure you have Python installed, then install the required dependencies:

pip install numpy matplotlib scikit-learn


🚀 How to Run

Run the script directly using Python:

python main.py


A window will display showing the scatter plot of Setosa (red) and Versicolor (blue) data points, with a dashed black boundary line animating across 50 epochs until it accurately divides the two classes.

🔍 Code Explanation

1. Data Preparation

Filters out Virginica samples (target < 2).

Selects the first two features ($x_1$ = Sepal Length, $x_2$ = Sepal Width).

Standardizes feature dimensions using $z = \frac{X - \mu}{\sigma}$ to ensure smooth gradient updates.

2. Training & Tracking Loop

Randomly initializes $w_1, w_2$ and sets $b = 0$.

Runs Gradient Descent for 50 epochs.

Appends (weights, bias) to a history list after each epoch.

3. Animated Rendering

Unscales standardized parameters back into raw Sepal Length / Sepal Width units for real-world plot axes.

Uses matplotlib.animation.FuncAnimation to re-render the decision line frame-by-frame at 100ms intervals.