# Logistic Regression Theory

## Historical Context
Logistic regression has its roots in the 19th century, first introduced by statistician Pierre François Verhulst in 1838 when studying population growth. The modern form was developed by Joseph Berkson in 1944, who coined the term "logit." It gained prominence in machine learning during the 1970s and became fundamental for binary classification problems.

Key historical developments:
- 1838: Verhulst introduces the logistic function
- 1944: Berkson develops the logit model
- 1958: David Cox contributes to the statistical theory
- 1970s: Widespread adoption in machine learning

## Introduction
Logistic regression is a fundamental classification algorithm used for binary classification problems. Despite its name, it's a classification algorithm rather than a regression algorithm.

## Why Not Linear Regression for Classification?

Linear regression is unsuitable for classification tasks for several key reasons:

1. **Unbounded Output**
   - Linear regression outputs unbounded values (-∞ to +∞)
   - Classification requires probabilities (0 to 1)
   - Linear regression can give meaningless probability predictions (> 1 or < 0)

2. **Non-Linear Decision Boundaries**
   - Classification often requires non-linear decision boundaries
   - Linear regression assumes linear relationships
   - Class separation is typically better modeled by sigmoid function

3. **Statistical Assumptions**
   - Linear regression assumes normally distributed errors
   - Binary outcomes violate this assumption
   - Logistic regression properly models Bernoulli distribution

4. **Cost Function Issues**
   - Mean Squared Error (MSE) for binary outcomes is non-convex
   - Multiple local minima make optimization difficult
   - Log loss provides better optimization properties

## Types of Logistic Regression

### 1. Binary Logistic Regression
- Most common type
- Predicts a binary outcome (0 or 1)
- Examples: 
  - Email spam (spam/not spam)
  - Medical diagnosis (disease present/absent)
  - Customer churn (will churn/won't churn)

### 2. Multinomial Logistic Regression
- Handles multiple class outcomes
- Each class has its own set of parameters
- Uses softmax function instead of sigmoid
- Examples:
  - Document classification (multiple topics)
  - Product categorization
  - Image classification

### 3. Ordinal Logistic Regression
- For ordered categorical outcomes
- Maintains order relationship between classes
- Examples:
  - Movie ratings (1 to 5 stars)
  - Education levels (elementary, high school, college)
  - Customer satisfaction levels

## The Logistic Function (Sigmoid)
The core of logistic regression is the sigmoid function:
```
σ(z) = 1 / (1 + e^(-z))
```
This function maps any real number to a value between 0 and 1, making it perfect for binary classification.

## Mathematical Formulation

### 1. Hypothesis Function
Unlike linear regression's linear hypothesis, logistic regression uses:
```
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```
where:
- θ represents the model parameters
- x represents the input features
- θᵀx is the dot product of parameters and features

### 2. Cost Function
The cost function for logistic regression is:
```
J(θ) = -1/m ∑[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
```
where:
- m is the number of training examples
- y⁽ⁱ⁾ is the actual label (0 or 1)
- h_θ(x⁽ⁱ⁾) is the predicted probability

### 3. Gradient Descent
The gradient descent update rule is:
```
θⱼ := θⱼ - α * ∂J/∂θⱼ
```
where:
- α is the learning rate
- ∂J/∂θⱼ = 1/m ∑(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾ⱼ

## Detailed Understanding and Derivations

### The Sigmoid Function Explained
The sigmoid function σ(z) = 1/(1 + e^(-z)) has important properties:
1. **Bounded**: Output always between 0 and 1
2. **Monotonic**: Strictly increasing
3. **Differentiable**: Its derivative is σ(z)(1 - σ(z))

Derivation of sigmoid derivative:
```
Let y = σ(z) = 1/(1 + e^(-z))
dy/dz = -(1/(1 + e^(-z))^2) * (-e^(-z))
      = e^(-z)/(1 + e^(-z))^2
      = (1/(1 + e^(-z))) * (e^(-z)/(1 + e^(-z)))
      = y * (1 - y)
```

### Cost Function Derivation
The cost function is derived using Maximum Likelihood Estimation (MLE):

1. Probability of outcome given features:
   P(y|x;θ) = (h_θ(x))^y * (1-h_θ(x))^(1-y)

2. Likelihood function for m samples:
   L(θ) = ∏ᵢ₌₁ᵐ P(y⁽ⁱ⁾|x⁽ⁱ⁾;θ)

3. Log-likelihood (easier to work with):
   log L(θ) = ∑ᵢ₌₁ᵐ [y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]

4. Cost function (negative log-likelihood):
   J(θ) = -1/m * log L(θ)

### Gradient Descent Derivation
The partial derivative of the cost function:

1. Start with cost function J(θ)
2. Take partial derivative with respect to θⱼ:
```
∂J/∂θⱼ = -1/m ∑ [y⁽ⁱ⁾(1/h_θ(x⁽ⁱ⁾)) * ∂h_θ/∂θⱼ - (1-y⁽ⁱ⁾)/(1-h_θ(x⁽ⁱ⁾)) * ∂h_θ/∂θⱼ]
```
3. Using chain rule and sigmoid derivative:
```
∂h_θ/∂θⱼ = h_θ(x⁽ⁱ⁾)(1-h_θ(x⁽ⁱ⁾))x⁽ⁱ⁾ⱼ
```
4. Final gradient:
```
∂J/∂θⱼ = 1/m ∑(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾ⱼ
```

## Mathematical Derivation: A Deeper Look

### From Linear to Logistic
1. Start with linear model:
   ```
   z = θᵀx
   ```

2. Need transformation to get probabilities:
   ```
   P(y=1|x) = g(θᵀx)
   ```
   where g must map to [0,1]

3. Sigmoid function chosen because:
   - Natural S-shaped curve
   - Differentiable
   - Models "tipping point" behavior

### Maximum Likelihood Derivation

1. **Probability Model**
   For binary classification:
   ```
   P(y|x;θ) = (h_θ(x))^y * (1-h_θ(x))^(1-y)
   ```
   This represents Bernoulli distribution

2. **Likelihood Function**
   ```
   L(θ) = ∏ᵢ₌₁ᵐ P(y⁽ⁱ⁾|x⁽ⁱ⁾;θ)
        = ∏ᵢ₌₁ᵐ (h_θ(x⁽ⁱ⁾))^y⁽ⁱ⁾ * (1-h_θ(x⁽ⁱ⁾))^(1-y⁽ⁱ⁾)
   ```

3. **Log-Likelihood**
   Taking log for easier optimization:
   ```
   ℓ(θ) = log L(θ)
        = ∑ᵢ₌₁ᵐ [y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
   ```

4. **Cost Function**
   Negative log-likelihood (to minimize):
   ```
   J(θ) = -1/m * ℓ(θ)
   ```

### Gradient Descent Proof
Starting with ∂J/∂θⱼ:

1. **Chain Rule Application**
   ```
   ∂J/∂θⱼ = -1/m ∑ [y⁽ⁱ⁾/h_θ(x⁽ⁱ⁾) * ∂h_θ/∂θⱼ - (1-y⁽ⁱ⁾)/(1-h_θ(x⁽ⁱ⁾)) * ∂h_θ/∂θⱼ]
   ```

2. **Sigmoid Derivative**
   ```
   ∂h_θ/∂θⱼ = h_θ(x⁽ⁱ⁾)(1-h_θ(x⁽ⁱ⁾))x⁽ⁱ⁾ⱼ
   ```

3. **Substitution and Simplification**
   ```
   ∂J/∂θⱼ = 1/m ∑(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾ⱼ
   ```

This derivation proves:
- Cost function is convex
- Global minimum exists
- Gradient descent will converge

## Key Differences from Linear Regression

1. **Output Range**: 
   - Linear Regression: (-∞, +∞)
   - Logistic Regression: [0,1]

2. **Hypothesis Function**:
   - Linear: h_θ(x) = θᵀx
   - Logistic: h_θ(x) = σ(θᵀx)

3. **Cost Function**:
   - Linear: Mean Squared Error
   - Logistic: Log Loss (Cross-Entropy)

## Model Interpretation

### Decision Boundary Analysis
The decision boundary occurs where:
```
θᵀx = 0
h_θ(x) = 0.5
```

For multiple features:
- Linear boundary: θ₀ + θ₁x₁ + θ₂x₂ = 0
- Polynomial features create non-linear boundaries

### Probability Interpretation
- h_θ(x) represents P(y=1|x;θ)
- 1-h_θ(x) represents P(y=0|x;θ)
- Log odds (logit): log(P(y=1|x;θ)/(1-P(y=1|x;θ))) = θᵀx

## Regularization in Logistic Regression

### L2 Regularization
Modified cost function:
```
J(θ) = -1/m ∑[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))] + λ/2m ∑θⱼ²
```
where λ is the regularization parameter.

### L1 Regularization
```
J(θ) = -1/m ∑[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))] + λ/m ∑|θⱼ|
```

## Implementation Considerations

1. **Feature Scaling**
   - Important for faster convergence
   - Normalize features to similar ranges

2. **Learning Rate**
   - Too large: May not converge
   - Too small: Slow convergence

3. **Decision Boundary**
   - Predict 1 if h_θ(x) ≥ 0.5
   - Predict 0 if h_θ(x) < 0.5

## Optimization Techniques

### Beyond Basic Gradient Descent
1. **Stochastic Gradient Descent (SGD)**
   - Updates parameters using single examples
   - Faster convergence for large datasets

2. **Mini-batch Gradient Descent**
   - Compromise between batch and stochastic
   - Batch size typically 32-256

3. **Newton's Method**
   - Uses second-order derivatives
   - Faster convergence but computationally expensive
```
θ := θ - H⁻¹∇J(θ)
```
where H is the Hessian matrix.

## Advantages and Limitations

### Advantages
- Simple and efficient
- Less prone to overfitting with small datasets
- Easily interpretable results
- Provides probability scores

### Limitations
- Assumes linear decision boundary
- Requires more data for stable results
- May underperform with complex relationships
- Sensitive to outliers

## Applications
- Spam Detection
- Medical Diagnosis
- Credit Risk Assessment
- Marketing Response Prediction
