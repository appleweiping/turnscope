# Conversation classifier CLI

TurnScope exposes its dependency-free multinomial Naive Bayes model through a
reproducible command-line boundary:

```console
turnscope classify train.jsonl predict.json \
  --labels labels.json --model classifier.json --output predictions.json
```

`labels.json` maps every training conversation ID to a non-empty string label.
The command emits a model digest and one prediction with normalized class
probabilities per input conversation. The optional model artifact is
authenticated by its canonical digest and can be loaded through
`ConversationClassifier.load()` for later deployment. Training and prediction
inputs, outputs, and model artifacts are refused when they alias one another.
