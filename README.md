# Forsaken-Apex-Evaluation

## Open-Set Detection

The model was trained exclusively on the 8 known defect classes present in the training dataset. No additional class was introduced into the model architecture or training procedure to represent novel or unseen defect types.

To handle test samples belonging to defect categories outside the training distribution (Crack, LER, Open, Other), we applied a **confidence-based rejection threshold** at inference time. After generating softmax probabilities across the 8 known classes, any prediction where the maximum class confidence fell below a threshold of `0.35` was assigned to an `other` bucket, indicating that the model was not sufficiently confident in any known-class prediction.

```
if max(softmax_probabilities) < 0.35:
    prediction = "other"
```

This approach requires no architectural changes and adds no learnable parameters. It operates purely as a post-processing step on the model's output probabilities.

### Limitations

This method does not constitute true open-set recognition. The model has no explicit representation of unseen defect types and cannot distinguish between them. A novel defect that visually resembles a known class may still be confidently predicted as that known class and will not be caught by the threshold. In the evaluation set, 141 of 296 samples belonged to unseen classes, of which 56 were correctly rejected and 85 were misclassified as a known defect type.
