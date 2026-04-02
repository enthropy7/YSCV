# yscv-recognize

Face and object recognition with embedding matching and VP-Tree approximate nearest neighbor search.

```rust
use yscv_recognize::*;

let mut recognizer = Recognizer::new();
recognizer.enroll("alice", &embedding_alice);
recognizer.enroll("bob", &embedding_bob);

let matches = recognizer.identify(&query_embedding, 0.7);
// matches[0].label == "alice", matches[0].score == 0.95
```

## Features

- **Cosine similarity** matching between embeddings
- **VP-Tree** (Vantage Point Tree) for fast approximate nearest neighbor
- **Gallery management**: enroll, remove, snapshot, restore
- **Serializable**: save/load gallery as JSON

## Tests

16 tests covering enrollment, matching, gallery persistence.
