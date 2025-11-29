import numpy as np


def generate_logistic_data(num_samples: int):
    w_true = 3.5
    b_true = -1.0

    xs = np.random.uniform(-5, 5, size=num_samples).astype(np.float32)

    z = w_true * xs + b_true
    p = 1.0 / (1.0 + np.exp(-z))

    ys = (np.random.rand(num_samples).astype(np.float32) < p).astype(np.float32)
    return xs, ys, w_true, b_true


def train_logistic_cpu(num_samples=2000, num_epochs=2000, lr=0.05):
    xs, ys, w_true, b_true = generate_logistic_data(num_samples)

    w = 0.0
    b = 0.0
    N = float(num_samples)

    for epoch in range(num_epochs):
        z = w * xs + b
        y_pred = 1.0 / (1.0 + np.exp(-z))
        e = y_pred - ys

        grad_w = (1.0 / N) * np.sum(e * xs)
        grad_b = (1.0 / N) * np.sum(e)

        w -= lr * grad_w
        b -= lr * grad_b

        if (epoch + 1) % 200 == 0:
            eps = 1e-7
            loss = -np.mean(
                ys * np.log(y_pred + eps) + (1 - ys) * np.log(1 - y_pred + eps)
            )
            print(
                f"Epoch {epoch+1}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}, "
                f"true_w={w_true}, true_b={b_true}"
            )

    return w, b, w_true, b_true


if __name__ == "__main__":
    w, b, w_true, b_true = train_logistic_cpu()
    print("Final:")
    print("  learned w:", w)
    print("  learned b:", b)
    print("  true w   :", w_true)
    print("  true b   :", b_true)