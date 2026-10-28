# Local artifacts

Owner: Member 6 coordinates storage; each model owner supplies its artifacts.
`models/` is empty and ignored apart from its directory marker. Do not commit
weights, checkpoints, serialized estimators, or downloaded pretrained models.

Future handoffs need a versioned artifact location, SHA-256, model family and
mode, ordered labels, input/preprocessing contract version, feature settings,
training split hash, seed, code commit, dependencies, and load instructions.
No artifact can currently be loaded because model I/O is not implemented.
