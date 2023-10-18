
library("neuralnet")
library('caret')
library("gplots")



dataset <- read.csv("creditset.csv")
head(dataset)
print("---------------------------------------------")


plot(dataset$age, dataset$LTI, col = ifelse(dataset$default10yr == 1, "red", "blue"),
     xlab = "Age", ylab = "LTI", main = "Scatterplot of Age vs. LTI")

legend("topright", legend = c("Not Defaulted", "Defaulted"),
       col = c("blue", "red"), pch = 1, bg = "white")





# extract a set to train the NN
trainset <- dataset[1:1600, ]

# select the test set
testset <- dataset[1601:2000, ]

# build the neural network (NN)

creditnet <- neuralnet(default10yr ~ LTI + age, trainset, hidden = c(3,3), lifesign = "minimal", 
                       linear.output = FALSE, threshold = 0.1)

# creditnet <- neuralnet(default10yr ~ LTI + age, trainset, hidden = 4, lifesign = "minimal", 
#     linear.output = FALSE, threshold = 0.1)





## test the resulting output
temp_test <- subset(testset, select = c("LTI", "age"))
head(temp_test)
print("---------------------------------------------")

creditnet.results <- compute(creditnet, temp_test)
head(creditnet.results)


results <- data.frame(actual = testset$default10yr, prediction = creditnet.results$net.result)
head(results)
print("---------------------------------------------")

results$prediction <- round(results$prediction)
head(results)
# results[1:15, ]
print("---------------------------------------------")

accuracy <- sum(results$prediction == results$actual) / nrow(results)
cat("Accuracy:", accuracy, "\n")
print("---------------------------------------------")

CM <- confusionMatrix(data=factor(results$prediction), reference = factor(results$actual))
CM
print("---------------------------------------------")

# Create a histogram of actual counts and predicted probabilities side by side
par(mfrow = c(1, 2))  # Set up a 1x2 grid for plots

# Plot histogram of actual counts
hist(results$actual, breaks = 2, main = "Histogram of Actual Counts", 
     xlab = "Actual", ylab = "Frequency", col = "blue")

# Plot histogram of predicted probabilities
hist(results$prediction, breaks = 2, main = "Histogram of Predicted Probabilities", 
     xlab = "Predicted Probability", ylab = "Frequency", col = "red")

par(mfrow = c(1, 1))  # Reset the plotting layout to the default

## plot the NN
plot(creditnet, rep = "best")





