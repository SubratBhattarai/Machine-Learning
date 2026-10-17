# 1.-Perceptron-
Pure Numpy demonstration of Perceptron for AND, OR, NAND, and how it fails for NOR gate.
# Perceptron: Learning AND, OR, NAND, and NOR Logic Gates

## 📌 Project Overview

This project implements a **single-layer Perceptron** from scratch to understand how a simple artificial neuron learns logical operations.

A perceptron is one of the earliest and simplest models in machine learning. It takes numerical inputs, multiplies them by weights, adds a bias, and applies an activation function to produce a binary output.

In this project, the perceptron is used to explore four fundamental Boolean logic gates:

- AND
- OR
- NAND
- NOR

The goal is to understand how a perceptron learns decision boundaries and why its ability to learn a logic gate depends on the training data, weights, bias, and model configuration.

## 🧠 What Is a Perceptron?

A perceptron is a binary classifier that calculates a weighted sum of its inputs and passes the result through a step activation function.

### Mathematical Formula

The weighted sum is:

\[
z = w_1x_1 + w_2x_2 + b
\]

The predicted output is:

\[
\hat{y} =
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
\]

Where:

- \(x_1, x_2\): Input values
- \(w_1, w_2\): Weights assigned to the inputs
- \(b\): Bias term
- \(z\): Weighted sum
- \(\hat{y}\): Predicted binary output

During training, the perceptron adjusts its weights and bias based on prediction errors.

## 🔢 Boolean Logic Gates

Each logic gate receives two binary inputs and produces a binary output.

### 1. AND Gate

The AND gate returns `1` only when **both inputs are 1**.

| Input 1 | Input 2 | Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**Expected behavior:** The perceptron learns to classify only the input combination `(1, 1)` as positive.

### 2. OR Gate

The OR gate returns `1` when **at least one input is 1**.

| Input 1 | Input 2 | Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

**Expected behavior:** The perceptron learns to classify every combination except `(0, 0)` as positive.

### 3. NAND Gate

NAND means NOT AND. It returns `0` only when both inputs are `1`.

| Input 1 | Input 2 | Output |
|---:|---:|---:|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Expected behavior:** The perceptron learns the opposite output of the AND gate.

### 4. NOR Gate

NOR means NOT OR. It returns `1` only when **both inputs are 0**.

| Input 1 | Input 2 | Output |
|---:|---:|---:|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |

**Expected behavior:** The perceptron must classify `(0, 0)` as positive and all other combinations as negative.

## ⚙️ Why Does the Perceptron Fail for NOR?

A single-layer perceptron is capable of learning the NOR gate because NOR is **linearly separable**.

Therefore, if the perceptron fails to learn NOR in this project, the failure is not a fundamental limitation of the perceptron architecture.

Possible causes include:

1. **Incorrect labels:** The target outputs may not match the NOR truth table.
2. **Incorrect bias or activation threshold:** The model may classify zero-valued inputs incorrectly.
3. **Training configuration:** The learning rate, number of epochs, or weight initialization may prevent convergence within the chosen training process.
4. **Implementation errors:** The weight update rule, prediction function, or bias update may be incorrect.
5. **Incorrect model setup:** The code may reuse weights or other settings from a different logic gate instead of training a separate model.

### Important Mathematical Insight

AND, OR, NAND, and NOR are all linearly separable. A single-layer perceptron can represent all four gates.

For example, a valid NOR configuration is:

- \(w_1 = -1\)
- \(w_2 = -1\)
- \(b = 0.5\)

Using the rule that the output is `1` when \(z > 0\), the model produces the correct NOR outputs for all four input combinations.

This demonstrates that a NOR failure should be investigated as a training or implementation issue rather than an inherent limitation of the perceptron.

## 📐 Linear Separability

A dataset is linearly separable if a straight line can separate its two output classes.

For two inputs, the perceptron creates a decision boundary:

\[
w_1x_1 + w_2x_2 + b = 0
\]

All four logic gates in this project can be represented using a linear decision boundary.

This is