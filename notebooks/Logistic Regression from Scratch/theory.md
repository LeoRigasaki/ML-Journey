# Logistic Regression Theory: A Comprehensive Guide

In my previous notes, I explained Linear Regression and how it works. Let's explore why Logistic Regression is one of the important topics to understand.

## Table of Contents
1. What is Logistic Regression?
2. Types of Logistic Regression
3. Assumptions of Logistic Regression
4. Why not Linear Regression for Classification?
5. The Logistic Model
6. Interpretation of the coefficients
7. Odds Ratio and Logit
8. Decision Boundary
9. Cost Function of Logistic Regression
10. Gradient Descent in Logistic Regression
11. Evaluating the Logistic Regression Model

Let's get started!

![Photo by Dose Media on Unsplash]

## Introduction
Logistic regression is a supervised statistical technique to find the probability of dependent variable (classes present in the variable). It uses logit functions to derive relationships between dependent and independent variables by predicting probabilities of occurrence. The logistic functions (sigmoid functions) convert probabilities into binary values for predictions. Pierre François Verhulst introduced the logistic function in the 19th century. It gained prominence in:

## Types of Logistic Regression
### 1. Binary Logistic Regression
- Only two possible outcomes/classes
- Most common implementation
- Example: Male/Female classification

### 2. Multinomial Logistic Regression
- Three or more possible outcomes without ordering
- Uses softmax function instead of sigmoid
- Example: Food quality prediction (Good/Great/Bad)

### 3. Ordinal Logistic Regression
- Three or more possible outcomes with ordering
- Example: Star ratings (1-5 stars)

## Assumptions of Logistic Regression
Unlike Linear Regression, Logistic Regression:
- Does not require linear relationship between dependent and independent variables
- Error terms do not need to be normally distributed
- Homoscedasticity is not required

Key assumptions include:
1. **Minimal Multicollinearity**: Little or no multicollinearity among independent variables
   - Can be checked using VIF (Variance Inflation Factor)

2. **Linear Relationship**: Independent variables should be linearly related to log odds
   - Verified using Box-Tidwell test

3. **Sample Size**: Requires large sample for good predictions

4. **Independence**: Observations must be independent of each other

5. **No Influential Values**: Absence of significant outliers in continuous predictors
   - Can be checked using IQR, z-score, or box/violin plots

## Why Not Linear Regression for Classification?
Linear Regression is unsuitable because:
1. **Output Range**: 
   - Linear regression predicts continuous values (-∞ to +∞)
   - Classification needs probabilities (0 to 1)

2. **Threshold Issues**:
   - Hard to find right threshold for class distinction
   - Particularly problematic for multiclass problems

3. **Class Ordering**:
   - Forces numerical ordering on categorical classes
   - Creates artificial relationships between classes

4. **Continuous Output**:
   - Produces values outside valid class range
   - Best fit line passes through mean of points

## The Logistic Model
The logistic model addresses these issues by:
1. Condensing output between 0 and 1
2. Using sigmoid function where:
   - b₀ + b₁X = 0 → p = 0.5
   - b₀ + b₁X > 0 → p approaches 1
   - b₀ + b₁X < 0 → p approaches 0

## Interpretation and Odds Ratio

### Coefficient Interpretation
- Unlike linear regression, coefficients represent change in log odds
- Rate of change is non-linear due to sigmoid function

### Odds Ratio
Odds = P(success) / P(failure)
- Range: 0 to ∞
- Example:
  - If P(success) = 0.6
  - P(failure) = 0.4
  - Odds(success) = 0.6/0.4 = 1.5
  - Odds(failure) = 0.4/0.6 = 0.67

### Logit Function
- Log of odds ratio
- Transforms probability space to real number space
- Formula: odds = p/(1-p)

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

$J(θ) = -1/m ∑[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]$

where:
- m is the number of training examples
- y⁽ⁱ⁾ is the actual label (0 or 1)
- h_θ(x⁽ⁱ⁾) is the predicted probability

### 3. Gradient Descent
The gradient descent update rule is:

$θⱼ := θⱼ - α * ∂J/∂θⱼ$

where:
- $α$ is the learning rate
- $∂J/∂θⱼ = 1/m ∑(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾ⱼ$

## Detailed Understanding and Derivations
### The Sigmoid Function Explained
The sigmoid function σ(z) = 1/(1 + e^(-z)) has important properties:
1. **Bounded**: Output always between 0 and 1
2. **Monotonic**: Strictly increasing
3. **Differentiable**: Its derivative is $σ(z)(1 - σ(z))$

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
   ```
   P(y|x;θ) = (h_θ(x))^y * (1-h_θ(x))^(1-y)
   ```

2. Likelihood function for m samples:
   ```
   L(θ) = ∏ᵢ₌₁ᵐ P(y⁽ⁱ⁾|x⁽ⁱ⁾;θ)
        = ∏ᵢ₌₁ᵐ (h_θ(x⁽ⁱ⁾))^y⁽ⁱ⁾ * (1-h_θ(x⁽ⁱ⁾))^(1-y⁽ⁱ⁾)
   ```

3. Log-likelihood (easier to work with):
   ```
   log L(θ) = ∑ᵢ₌₁ᵐ [y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
   ```

4. Cost function (negative log-likelihood):
   ```
   J(θ) = -1/m * log L(θ)
   ```

### Gradient Descent Derivation
The partial derivative of the cost function:
```
∂J/∂θⱼ = -1/m ∑ [y⁽ⁱ⁾/h_θ(x⁽ⁱ⁾) * ∂h_θ/∂θⱼ - (1-y⁽ⁱ⁾)/(1-h_θ(x⁽ⁱ⁾)) * ∂h_θ/∂θⱼ]
```
Using chain rule and sigmoid derivative:
```
∂h_θ/∂θⱼ = h_θ(x⁽ⁱ⁾)(1-h_θ(x⁽ⁱ⁾))x⁽ⁱ⁾ⱼ
```
Substitution and simplification:
```
∂J/∂θⱼ = 1/m ∑(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾ⱼ
```

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

## Model Evaluation

### R² (R-Squared) Value
- Different from Linear Regression's R²
- Uses Pseudo R² to measure predictive power
- McFadden's R² is generally preferred
- Multiple approaches available, no universal best method

### AIC (Akaike Information Criteria)
- Estimates model's goodness-of-fit
- Measures information loss
- Lower AIC indicates better model
- Not affected by adding variables
- Useful for model selection

The AIC formula is:
```
AIC = -2/N * LL + 2*K/N
```
Where:
- N = number of training samples
- LL = Log Likelihood of model
- K = number of parameters

### Additional Evaluation Methods
- Confusion Matrix
- ROC-AUC Curve
- See related blog: "Calculating Accuracy of an ML Model and Understanding the AUC-ROC Curve
