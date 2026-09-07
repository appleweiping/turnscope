# Conversation classifier

`ConversationClassifier` is a deterministic multinomial Naive Bayes baseline for
conversation-level labels. It operates on validated `Conversation` values and is
useful for checking a feature pipeline before introducing a larger model
dependency.

```python
from turnscope import ConversationClassifier

model = ConversationClassifier(max_features=20_000)
model.fit(conversations, {conversation.id: label for conversation, label in training})
label = model.predict(conversation)
probabilities = model.predict_proba(conversation)
model.save("classifier.json")
restored = ConversationClassifier.load("classifier.json")
```

Training uses Laplace-smoothed token likelihoods. The artifact includes a digest
over hyperparameters and fitted state; loading rejects a digest mismatch. This
is a transparent baseline, not a claim of neural-model equivalence.
