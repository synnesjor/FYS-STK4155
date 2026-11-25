from astroNN.datasets import galaxy10
from sklearn.model_selection import train_test_split
import os
import numpy as np
import time
import jax
import jax.numpy as jnp
from jax import lax
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# To load images and labels (will download automatically at the first time)
# First time downloading location will be ~/.astroNN/datasets/
images, labels = galaxy10.load_data()

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

def preprocess(imgs, size=64):
    """Downsample (N, 256, 256, 3) -> (N, size, size, 3) by block-averaging, then flatten."""
    N, H, W, C = imgs.shape
    f = H // size                                   # e.g. 256 // 32 = 8
    small = imgs.reshape(N, size, f, size, f, C).mean(axis=(2, 4))   # (N, size, size, 3)

    return small

small_train_images = preprocess(train_images)
small_test_images = preprocess(test_images)

def keras_cnn(train_inputs, train_labels, test_inputs, test_labels, epochs=5, batch=64, gamma=1e-3, seed=1, verbose=True):
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)
    trunk = [keras.Input(shape=(64, 64, 3)),
             layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D((2, 2)),
             layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D((2, 2))]

    head = [layers.GlobalAveragePooling2D(), layers.Dense(10, activation="softmax")]
    m = keras.Sequential(trunk + head)
    m.compile(optimizer=keras.optimizers.Adam(gamma), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
    n_par = int(sum(np.prod(w.shape) for w in m.trainable_weights))
    hist, t0 = [], time.time()
    for ep in tqdm(range(1, epochs + 1)):
        h = m.fit(train_inputs, train_labels, epochs=1, batch_size=batch, verbose=0)
        _, acc = m.evaluate(test_inputs, test_labels, verbose=0)
        hist.append((ep, h.history["loss"][0], acc, time.time() - t0))
        if verbose:
            print(f"  keras CNN epoch {ep}: loss {hist[-1][1]:.4f}  test accuracy {acc:.4f}  ({time.time() - t0:.0f} s)")
    layer_params = [(l.name, l.count_params()) for l in m.layers]
    return {"history": np.array(hist), "n_params": n_par, "test_accuracy": hist[-1][2],
            "seconds_per_epoch": hist[-1][3] / epochs, "layer_params": layer_params, "model": m}

#keras_cnn_results = keras_cnn(small_train_images, train_labels, small_test_images, test_labels, epochs=100, batch=64, gamma=1e-3, seed=1, verbose=True)

def pad2d(X, P):
    """Zero-pad the two spatial axes of X with shape (N, C, H, W)."""
    if P == 0:
        return X
    return np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode="constant")


def im2col(X, F, S, P):
    """Rearrange every F x F patch of X into a column, Eq. (10.34).

    X has shape (N, C, H, W); the result has shape (N, C*F*F, H2*W2) with
    H2 = (H - F + 2P)/S + 1, so that a convolution becomes one matrix product.
    """
    N, C, H, W = X.shape
    H2 = (H - F + 2 * P) // S + 1
    W2 = (W - F + 2 * P) // S + 1
    Xp = pad2d(X, P)
    cols = np.empty((N, C * F * F, H2 * W2))
    for i in range(F):
        for j in range(F):
            patch = Xp[:, :, i:i + S * H2:S, j:j + S * W2:S]      # (N,C,H2,W2)
            cols[:, (i * F + j)::F * F, :] = patch.reshape(N, C, -1)
    return cols


def col2im(cols, X_shape, F, S, P):
    """Adjoint of im2col: scatter columns back, accumulating overlaps."""
    N, C, H, W = X_shape
    H2 = (H - F + 2 * P) // S + 1
    W2 = (W - F + 2 * P) // S + 1
    Xp = np.zeros((N, C, H + 2 * P, W + 2 * P))
    for i in range(F):
        for j in range(F):
            patch = cols[:, (i * F + j)::F * F, :].reshape(N, C, H2, W2)
            np.add.at(Xp, (slice(None), slice(None),
                           slice(i, i + S * H2, S), slice(j, j + S * W2, S)), patch)
    return Xp if P == 0 else Xp[:, :, P:-P, P:-P]


def conv_forward(X, W, b, S=1, P=0):
    """Cross-correlation, Eq. (10.16).  W has shape (K, C, F, F)."""
    N, C, H, Wd = X.shape
    K, _, F, _ = W.shape
    H2 = (H - F + 2 * P) // S + 1
    W2 = (Wd - F + 2 * P) // S + 1
    cols = im2col(X, F, S, P)                       # (N, C*F*F, H2*W2)
    out = np.einsum("kd,ndp->nkp", W.reshape(K, -1), cols) + b[None, :, None]
    return out.reshape(N, K, H2, W2), cols


def conv_backward(dY, X, W, cols, S=1, P=0):
    """Gradients of the convolution, Eqs. (10.29), (10.30), (10.31)."""
    N, K, H2, W2 = dY.shape
    _, C, F, _ = W.shape
    dYf = dY.reshape(N, K, -1)                                   # (N,K,H2W2)
    dW = np.einsum("nkp,ndp->kd", dYf, cols).reshape(W.shape)
    db = dYf.sum(axis=(0, 2))
    dcols = np.einsum("kd,nkp->ndp", W.reshape(K, -1), dYf)
    dX = col2im(dcols, X.shape, F, S, P)
    return dX, dW, db


def maxpool_forward(X, F=2, S=2):
    """Max pooling, Eq. (10.21).  Returns the output and an argmax mask."""
    N, C, H, W = X.shape
    H2, W2 = (H - F) // S + 1, (W - F) // S + 1
    patches = np.empty((N, C, H2, W2, F * F))
    for i in range(F):
        for j in range(F):
            patches[..., i * F + j] = X[:, :, i:i + S * H2:S, j:j + S * W2:S]
    idx = patches.argmax(axis=-1)
    out = np.take_along_axis(patches, idx[..., None], axis=-1)[..., 0]
    return out, idx


def maxpool_backward(dY, X, idx, F=2, S=2):
    """Route each gradient to the argmax that produced it, Eq. (10.33)."""
    N, C, H, W = X.shape
    H2, W2 = dY.shape[2], dY.shape[3]
    dX = np.zeros_like(X)
    for i in range(F):
        for j in range(F):
            mask = (idx == i * F + j)
            np.add.at(dX, (slice(None), slice(None),
                           slice(i, i + S * H2, S), slice(j, j + S * W2, S)),
                      dY * mask)
    return dX

def relu_c(Z):
    return np.maximum(0.0, Z)


def relu_prime_c(Z):
    return (Z > 0).astype(Z.dtype)


def softmax_c(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy_c(probs, y):
    log_probs = jnp.log(jnp.maximum(probs, 1e-12))
    y = jnp.asarray(y)
    if y.ndim == 1:
        return -jnp.mean(jnp.take_along_axis(log_probs, y.astype(jnp.int32)[:, None], axis=1))
    return -jnp.mean(jnp.sum(y * log_probs, axis=1))


def init_cnn(C_in=1, K1=8, K2=16, F=3, n_out=10, flat=None, rng=None):
    """He initialisation, Eq. (8.74), with fan-in C*F*F for a convolution kernel."""
    rng = np.random.default_rng(0) if rng is None else rng

    def he(shape, fan_in):
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)
    return {
        "W1": he((K1, C_in, F, F), C_in * F * F), "b1": np.zeros(K1),
        "W2": he((K2, K1, F, F), K1 * F * F),     "b2": np.zeros(K2),
        "W3": he((flat, n_out), flat),            "b3": np.zeros(n_out),
    }


def _cnn_logits(p, X):
    """CNN logits for NCHW inputs, using JAX operations that support autodiff."""
    Z1 = lax.conv_general_dilated(
        X, p["W1"], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    ) + p["b1"][None, :, None, None]
    A1 = jax.nn.relu(Z1)
    P1 = lax.reduce_window(
        A1, -jnp.inf, lax.max, window_dimensions=(1, 1, 2, 2),
        window_strides=(1, 1, 2, 2), padding="VALID",
    )
    Z2 = lax.conv_general_dilated(
        P1, p["W2"], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    ) + p["b2"][None, :, None, None]
    A2 = jax.nn.relu(Z2)
    P2 = lax.reduce_window(
        A2, -jnp.inf, lax.max, window_dimensions=(1, 1, 2, 2),
        window_strides=(1, 1, 2, 2), padding="VALID",
    )
    flat = P2.reshape((P2.shape[0], -1))
    return flat @ p["W3"] + p["b3"]


def cnn_forward(p, X):
    """Eq. (10.35): conv-ReLU-pool-conv-ReLU-pool-dense-softmax."""
    X = jnp.asarray(X, dtype=jnp.float32)
    return jax.nn.softmax(_cnn_logits(p, X), axis=1)


def _cnn_loss(p, X, Y):
    log_probs = jax.nn.log_softmax(_cnn_logits(p, X), axis=1)
    Y = jnp.asarray(Y)
    if Y.ndim == 1:
        labels = Y.astype(jnp.int32)
        return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=1))
    if Y.ndim == 2 and Y.shape == log_probs.shape:
        return -jnp.mean(jnp.sum(Y * log_probs, axis=1))

# Define compiled JIT function.
loss_and_grad = jax.jit(jax.value_and_grad(_cnn_loss, argnums=0))

def cnn_backward(p, X, Y, return_loss=False):
    """Use JAX autodiff to calculate gradients of the CNN cross-entropy loss."""
    loss, gradients = loss_and_grad(p, jnp.asarray(X, dtype=jnp.float32), jnp.asarray(Y))
    return (loss, gradients) if return_loss else gradients


def cnn_accuracy(p, X, y, batch=128):
    pred = np.concatenate([np.array(cnn_forward(p, X[s:s + batch])).argmax(axis=1) for s in range(0, len(X), batch)])
    return float((pred == y).mean())


def cnn_predict(p, X, batch=128):
    return np.concatenate([np.array(cnn_forward(p, X[s:s + batch])).argmax(axis=1) for s in range(0, len(X), batch)])


def n_params(p):
    return sum(v.size for v in p.values())


def train_cnn(p, train_inputs, train_labels, test_inputs, test_labels, epochs=20, batch=32, gamma=3e-3, rng=None, verbose=True):
    """Adam on minibatches using JAX gradients; record loss, test accuracy, and elapsed time."""
    rng = np.random.default_rng(0) if rng is None else rng
    p = {k: jnp.asarray(val, dtype=jnp.float32) for k, val in p.items()}
    m = {k: jnp.zeros_like(val) for k, val in p.items()}
    v = {k: jnp.zeros_like(val) for k, val in p.items()}
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    hist, t0 = [], time.time()
    for ep in tqdm(range(1, epochs + 1)):
        order = rng.permutation(len(train_inputs))
        tot = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            batch_loss, g = cnn_backward(p, train_inputs[idx], train_labels[idx], return_loss=True)
            tot += float(batch_loss) * len(idx)
            t += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                p[k] = p[k] - gamma * (m[k] / (1 - b1 ** t)) / (jnp.sqrt(v[k] / (1 - b2 ** t)) + eps)
        acc = cnn_accuracy(p, test_inputs, test_labels)
        hist.append((ep, tot / len(train_inputs), acc, time.time() - t0))
        if verbose:
            print(f"  epoch {ep:3d}  train loss {tot / len(train_inputs):.4f}   test acc {acc:.4f}   {time.time() - t0:5.0f} s")
    return p, hist

small_train_images_nchw = np.transpose(small_train_images, (0, 3, 1, 2))
small_test_images_nchw = np.transpose(small_test_images, (0, 3, 1, 2))
flat_size = 64 * (small_train_images.shape[1] // 4) * (small_train_images.shape[2] // 4)
train_cnn_results = train_cnn(init_cnn(C_in=3, K1=32, K2=64, F=3, n_out=10, flat=flat_size), small_train_images_nchw, train_labels, small_test_images_nchw, test_labels, epochs=100, batch=64, gamma=3e-3, rng=np.random.default_rng(0), verbose=True)