# Transformer Architecture for F1 Telemetry Analysis

---

## Slide 1: Architecture Overview

### LapTransformer: Sequence-to-Scalar Regression

**Input:** Multi-channel telemetry sequences (B, C, N)
- Channels: Speed, Throttle, Brake, nGear, DRS, RPM, Distance, Curvature
- Sequence length: 2048 samples (distance-aligned)

**Architecture Flow:**
```
Input (B,C,N) → Conv1d Projection → Sinusoidal PE → 
Transformer Encoder (4 layers, 4 heads) → 
Global Pooling (Mean + Max) → MLP Head → Output (B,)
```

**Key Hyperparameters:**
- d_model: 128, nhead: 4, depth: 4, d_ff: 256
- ~200-300K parameters

**Output:** Predicted qualifying lap time (seconds)

---

## Slide 2: Key Components & Design

### Core Components

**1. Input Projection (Conv1d)**
- Maps C channels → d_model=128 embedding space
- Enables channel interaction learning

**2. Sinusoidal Positional Encoding**
- Fixed sinusoidal patterns for distance-aware processing
- Supports sequences up to 8192 tokens

**3. Transformer Encoder Stack (4 layers)**
- Multi-head self-attention (4 heads) captures long-range dependencies
- Feed-forward network (128→256→128) with ReLU
- Pre-LayerNorm architecture with residual connections

**4. Global Pooling**
- Mean pooling: captures overall lap characteristics
- Max pooling: captures critical moments (braking, acceleration)
- Concatenated: 256-dim feature vector

**5. Prediction Head**
- 2-layer MLP: 256 → 128 → 1
- LayerNorm + ReLU + Dropout(0.1)

### Design Choices
✅ Distance-aligned sequences (not time-aligned)  
✅ Z-scoring per weekend (track-specific normalization)  
✅ Encoder-only architecture (no decoder needed)  
✅ SmoothL1Loss + AdamW optimizer

---

# Presentation Script

## Slide 1: Architecture Overview

**[Opening]** Today I'll present the transformer architecture we've developed for predicting F1 qualifying lap times from practice session telemetry data.

**[Input Description]** Our model takes multi-channel telemetry sequences as input. Specifically, we have a batch of sequences where each sequence has shape (B, C, N). B is the batch size, C represents the number of channels - which includes telemetry features like Speed, Throttle, Brake, gear number, DRS status, and RPM, plus track features like Distance and Curvature. N is the sequence length, which we've set to 2048 samples. Importantly, these sequences are distance-aligned rather than time-aligned, meaning we resample the telemetry data to a fixed grid of 2048 points evenly spaced along the track distance. This ensures consistency when comparing different drivers or laps, regardless of their speed.

**[Architecture Flow]** The architecture follows a clear pipeline. First, the multi-channel input goes through a 1D convolution layer that projects all channels into a unified embedding space of dimension 128. This is followed by sinusoidal positional encoding, which adds distance-aware position information to each sample in the sequence. The encoded sequence then passes through a stack of 4 transformer encoder layers, each with 4 attention heads. After encoding, we perform global pooling - specifically, we take both the mean and maximum across the sequence dimension, which gives us a comprehensive representation of the entire lap. Finally, a multi-layer perceptron head maps this pooled representation to a single scalar output - the predicted qualifying lap time in seconds.

**[Hyperparameters]** The model uses relatively modest hyperparameters: an embedding dimension of 128, 4 attention heads, 4 encoder layers, and a feed-forward dimension of 256. This results in a lightweight model with approximately 200 to 300 thousand parameters, making it efficient to train and deploy.

**[Output]** The output is a simple scalar prediction - the best qualifying lap time in seconds for each driver in the batch.

---

## Slide 2: Key Components & Design

**[Component 1: Input Projection]** Let's dive deeper into each component. The input projection layer uses a 1D convolution with kernel size 1, which is essentially a pointwise convolution. This layer maps our multi-channel input - which could have 8 or more channels depending on what telemetry we include - into a unified 128-dimensional embedding space. This projection is crucial because it allows the model to learn how different telemetry channels interact with each other. For example, it can learn that high throttle combined with low speed might indicate a corner, or that DRS activation typically occurs at high speeds.

**[Component 2: Positional Encoding]** Next, we add sinusoidal positional encoding. This is a fixed, non-learnable encoding that uses sine and cosine functions with different frequencies to encode position information. The formula uses powers of 10,000 to create different frequency patterns. This encoding is added element-wise to our embeddings and provides the transformer with distance-aware information. Since transformers don't inherently understand sequence order, this encoding is essential. We've designed it to support sequences up to 8192 tokens, though we typically use 2048.

**[Component 3: Transformer Encoder]** The heart of our architecture is the transformer encoder stack. We use 4 identical encoder layers. Each layer contains two main sub-components: multi-head self-attention and a feed-forward network. The self-attention mechanism allows each position in the sequence to attend to all other positions, which is crucial for F1 data. For instance, the model can learn that how a driver exits turn 1 affects their approach to turn 2, or that braking points in one sector relate to acceleration in another. We use 4 attention heads, meaning we compute 4 different attention patterns in parallel, each operating on 32 dimensions. The feed-forward network is a 2-layer MLP that expands from 128 to 256 dimensions and back to 128, with ReLU activation. This provides non-linear transformations that allow the model to learn complex feature interactions. We use a Pre-LayerNorm architecture, meaning layer normalization happens before the attention and feed-forward operations, which has been shown to improve training stability.

**[Component 4: Global Pooling]** After the encoder stack, we have a sequence of 2048 position embeddings, each of dimension 128. To convert this to a fixed-size representation for our regression head, we use global pooling. Specifically, we compute both the mean and maximum across the sequence dimension. The mean pooling captures the overall characteristics of the lap - the average speed, average throttle usage, and so on. The max pooling captures critical moments - the maximum speed achieved, the hardest braking point, peak RPM, and other peak features. By concatenating both, we get a 256-dimensional feature vector that comprehensively represents the entire lap.

**[Component 5: Prediction Head]** Finally, the prediction head is a 2-layer MLP that maps our 256-dimensional pooled features to a single scalar output. The first layer projects from 256 to 128 dimensions, followed by ReLU activation and dropout for regularization. The second layer maps from 128 to 1, giving us our predicted lap time. We use LayerNorm before the first linear layer to stabilize training.

**[Design Choices]** Now let me explain some key design choices. First, we use distance-aligned sequences rather than time-aligned. This means we resample telemetry to a fixed grid of points evenly spaced along the track. This is crucial because it allows us to compare sequences from different drivers or different speeds in a consistent way. A fast driver and a slow driver will have their telemetry aligned at the same track positions, making comparisons meaningful.

Second, we perform z-scoring per weekend. This means we normalize each channel within each race weekend separately. This accounts for track-specific conditions - different tracks have different typical speeds, different weather conditions, and track evolution throughout the weekend. By normalizing within each weekend, we ensure the model focuses on relative performance rather than absolute values.

Third, we use an encoder-only architecture. Since we're doing regression from a sequence to a scalar, we don't need a decoder. This makes the architecture simpler and more efficient. The encoder is sufficient to extract all the relevant information from the telemetry sequence.

Finally, for training, we use SmoothL1Loss, which is a Huber loss that's robust to outliers - important when dealing with real-world telemetry data that might have anomalies. We use the AdamW optimizer with a learning rate of 3e-4 and weight decay of 1e-4 for regularization.

**[Closing]** This architecture effectively captures the complex relationships in F1 telemetry data, allowing us to predict qualifying performance from practice session data with good accuracy. The transformer's ability to model long-range dependencies is particularly valuable for understanding how different parts of a lap relate to each other.

