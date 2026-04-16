function Score = scores(RealOutput, PredictedOutput, threshold)

PredictedOutput = PredictedOutput >= threshold;

TP = sum(RealOutput == 1 & PredictedOutput == 1);
TN = sum(RealOutput == 0 & PredictedOutput == 0);
FP = sum(RealOutput == 0 & PredictedOutput == 1);
FN = sum(RealOutput == 1 & PredictedOutput == 0);

Score.TP = TP;
Score.TN = TN;
Score.FP = FP;
Score.FN = FN;

Score.Recall = TP / (TP + FN + eps);
Score.Precision = TP / (TP + FP + eps);
Score.Specificity = TN / (TN + FP + eps);
Score.Accuracy = (TP + TN) / (FN + FP + TP + TN + eps);

Score.F1 = (2 * Score.Precision * Score.Recall) / (Score.Precision + Score.Recall + eps);
end