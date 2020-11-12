import numpy as np
import matplotlib.pyplot as plt


# Vanila Transformer attention implementation
def att(q, k, v, normalize=True):
    l, d = q.shape
    normalizer = 1 / (d ** 0.5) if normalize else 1
    a = np.exp(q @ k.T * normalizer)
    d_inv = np.diag(1 / (a @ np.ones(l)))
    return d_inv @ a @ v


# Perfomer attention implementation using some random feature map phi
def att_hat(q, k, v, phi, normalize=True):
    l, d = q.shape
    normalizer = 1 / (d ** 0.25)
    q_prime = phi(q * normalizer)
    k_prime = phi(k * normalizer)
    a_hat = (q_prime @ k_prime.T)
    d_inv = np.diag(1 / (a_hat @ np.ones(l)))
    return d_inv @ a_hat @ v


# random feature map
def phi(h, fs, random_feats):
    return lambda x: (
        h(x)
        / np.sqrt(m)
        * np.concatenate(
            [f(np.einsum("...d,md->...m", x, random_feats)) for f in fs],
            axis=-1,
        )
    )


# Performer "sin/cos" attention
def sincos_att_hat(q, k, v, random_feats, normalize=True):
    def h(x):
        return np.exp(np.square(x).sum(axis=-1, keepdims=True) / 2)

    sin = lambda x: np.sin(2 * np.pi * x)
    cos = lambda x: np.cos(2 * np.pi * x)

    kernel = phi(h, [sin, cos], random_feats)
    return att_hat(q, k, v, kernel, normalize)


# Performer "positive" attention
def positive_att_hat(q, k, v, random_feats, normalize=True):
    def h(x):
        return np.exp(-np.square(x).sum(axis=-1, keepdims=True) / 2)

    kernel = phi(h, [np.exp], random_feats)
    return att_hat(q, k, v, kernel, normalize)



# generate IID Gaussian random features
def iid_gaussian(m, d):
    return np.random.normal(size=(m, d))


# generate orthogonal Gaussian random features
def orthogonal_gaussian(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = np.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = np.vstack(blocks)
    matrix /= np.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix


# mean squared error
def mse(a, b):
    return np.square(a - b).mean()

###############################################################################
# The rest is just experiments
##############################################################################

# sequence length and hidden dim
l = 1024 # TODO: increase to 4096, will take longer
d = 16

num_samples = 15

# random feature sizes to try
ms = np.arange(d, 200, 16)


# Experiment:
# Sin/Cos attention vs Positive attention

sincos = []
positive = []
temperature = 1.5

np.random.seed(0)
for m in ms:
    sincos.append([])
    positive.append([])

    for _ in range(num_samples):
        q = np.random.randn(l, d) * temperature
        k = np.random.randn(l, d) * temperature
        v = np.random.randn(l, d) * temperature

        att_true = att(q, k, v)

        random_feats = orthogonal_gaussian(m, d)
        sincos[-1].append(mse(att_true, sincos_att_hat(q, k, v, random_feats)))
        positive[-1].append(mse(att_true, positive_att_hat(q, k, v, random_feats)))

sincos = np.array(sincos)
positive = np.array(positive)


def plot_line(x, y, label):
    mean = y.mean(axis=1)
    std = y.std(axis=1)
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean+std, mean-std, alpha=0.1)


plt.figure(figsize=(5, 3), dpi=300)
plot_line(ms, sincos, "Sin/Cos")
plot_line(ms, positive, "Positive")
plt.yscale("log")
plt.ylim(1e-2, 1e8)
plt.ylabel("Output MSE")
plt.xlabel("Num. Features $R$")
plt.legend();
plt.savefig("trig_vs_positive.png", bbox_inches="tight")



# Experiment:
# IID vs Orthogonal random features

iid = []
ortho = []

temperature = 1.0

np.random.seed(0)
for m in ms:
    iid.append([])
    ortho.append([])

    for _ in range(num_samples):
        q = np.random.randn(l, d) * temperature
        k = np.random.randn(l, d) * temperature
        v = np.random.randn(l, d) * temperature

        att_true = att(q, k, v)

        ortho_feats = orthogonal_gaussian(m, d)
        iid_feats = iid_gaussian(m, d)
        ortho[-1].append(mse(att_true, sincos_att_hat(q, k, v, ortho_feats)))
        iid[-1].append(mse(att_true, sincos_att_hat(q, k, v, iid_feats)))

iid = np.array(iid)
ortho = np.array(ortho)

plt.figure(figsize=(5, 3), dpi=300)
plot_line(ms, iid, "IID")
plot_line(ms, ortho, "Orthogonal")
plt.yscale("log")
plt.ylabel("Output MSE")
plt.xlabel("Num. Features $R$")
plt.legend();
plt.savefig("iid_vs_ortho.png", bbox_inches="tight")


