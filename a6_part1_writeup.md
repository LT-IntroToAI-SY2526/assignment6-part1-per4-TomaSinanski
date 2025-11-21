# Assignment 6 Part 1 - Writeup

**Name:** Toma Sinanski  
**Date:** 11/21/25

---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation
What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

**YOUR ANSWER:**
The R² score indicates how much of the variation in test scores can be explained by hours studied. If it’s close to 1, the model fits well and hours studied strongly predict the score. If it’s close to 0, the model explains very little and is a weak predictor.

---

### Question 2: Mean Squared Error (MSE)
What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

**YOUR ANSWER:**
MSE shows how far the model’s predictions are from the actual test scores on average. A higher MSE means the model performs worse. We square the errors to make all values positive and to ensure larger errors have a bigger impact than small ones.

---

### Question 3: Model Reliability
Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:
- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

**YOUR ANSWER:**
I would trust it to a degree because the highest amount of study time in the dataset is 9.8 hours, although most values cluster around 7–8 hours. The model can predict beyond its training range, but those predictions are less reliable since it hasn’t actually seen data for that range.

---

## Part 2: Data Analysis

### Question 4: Relationship Description
Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:
- Strong or weak?
- Linear or non-linear?
- Positive or negative?

**YOUR ANSWER:**
The scatter plot shows a strong, linear, and positive relationship between hours studied and test scores. The only noticeable inconsistency appears near the 8–9 hour range where the scores vary more.

---

### Question 5: Real-World Limitations
What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**
1. Amount of sleep before the test
2. A student’s motivation level
3. Quality of studying, teaching, or study materials

---

## Part 3: Code Reflection

### Question 6: Train/Test Split
Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

**YOUR ANSWER:**
We split the data so we can evaluate how well the model performs on new, unseen examples. If the model were trained and tested on the same data, it might appear perfect simply because it memorized the training data, meaning it wouldn’t generalize well.

---

### Question 7: Most Challenging Part
What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

**YOUR ANSWER:**
The hardest part was completing the write-up and figuring out how Anaconda and the imports worked. I needed to learn what each tool did in order to fully understand the code and answer the questions.

---

## Part 4: Extending Your Learning

### Question 8: Future Applications
Describe one real-world problem you could solve with linear regression. What would be your:
- **Feature (X):**
- **Target (Y):**
- **Why this relationship might be linear:**

**YOUR ANSWER:**
A real-world example would be predicting the price of a used car from its mileage.
- Feature (X): mileage
- Target (Y): car price

As mileage increases, the value of the car generally decreases at a steady rate, making the relationship roughly linear.

---


## Grading Checklist (for your reference)

Before submitting, make sure you have:
- [ ] Completed all functions in `a6_part1.py`
- [ ] Generated and saved `scatter_plot.png`
- [ ] Generated and saved `predictions_plot.png`
- [ ] Answered all questions in this writeup with thoughtful responses
- [ ] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:
1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
