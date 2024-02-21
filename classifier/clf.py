import tensorflow as tf
from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import Lookahead, AdamW
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K

mixed_precision.set_global_policy("mixed_float16")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import numpy as np

from imblearn.over_sampling import RandomOverSampler

from gc import collect

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency

from matplotlib import pyplot as plt

import pandas as pd

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["PATH"] = (
    os.environ["PATH"]
    + ";"
    + r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64"
)
print(os.environ["PATH"])


def get_centralized_gradients(optimizer, loss, params):
    """Compute the centralized gradients.

    From: https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow

    This function is ideally not meant to be used directly unless you are building a custom optimizer, in which case you
    could point `get_gradients` to this function. This is a modified version of
    `tf.keras.optimizers.Optimizer.get_gradients`.

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.
        loss: Scalar tensor to minimize.
        params: List of variables.

    # Returns:
      A gradients tensor.

    # Reference:
        [Yong et al., 2020](https://arxiv.org/abs/2004.01461)
    """

    # We here just provide a modified get_gradients() function since we are trying to just compute the centralized
    # gradients at this stage which can be used in other optimizers.
    grads = []
    for grad in K.gradients(loss, params):
        grad_len = len(grad.shape)
        if grad_len > 1:
            axis = list(range(grad_len - 1))
            grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
        grads.append(grad)

    if None in grads:
        raise ValueError(
            "An operation has `None` for gradient. "
            "Please make sure that all of your ops have a "
            "gradient defined (i.e. are differentiable). "
            "Common ops without gradient: "
            "K.argmax, K.round, K.eval."
        )
    if hasattr(optimizer, "clipnorm") and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [
            tf.keras.optimizers.clip_norm(g, optimizer.clipnorm, norm) for g in grads
        ]
    if hasattr(optimizer, "clipvalue") and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]
    return grads


def centralized_gradients_for_optimizer(optimizer):
    """Create a centralized gradients functions for a specified optimizer.

    From: From: https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.

    # Usage:

    ```py
    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> opt.get_gradients = gctf.centralized_gradients_for_optimizer(opt)
    >>> model.compile(optimizer = opt, ...)
    ```
    """

    def get_centralized_gradients_for_optimizer(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_optimizer


# Code was adapted from:
# https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
# https://keras.io/examples/vision/mlp_image_classification/

# custom activation function
def custom_activation(output):

    logexpsum = tf.keras.backend.sum(
        tf.keras.backend.exp(output), axis=-1, keepdims=True
    )
    result = logexpsum / (logexpsum + 1.0)
    return result


def select_supervised_samples(dataset, n_samples=64, n_classes=4):

    X, y = dataset

    X_list, _, y_list, _ = train_test_split(
        X, y, train_size=n_samples, shuffle=True, stratify=y
    )

    return np.asarray(X_list), np.asarray(y_list)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=4):
    # generate points in the latent space
    z_input = np.random.randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)

    return z_input


def generate_real_samples(dataset, n_samples, n_classes=4, n_per_class=5):

    X, y = dataset

    X_list, _, y_list, _ = train_test_split(
        X, y, train_size=n_samples, shuffle=True, stratify=y
    )

    y = np.ones((n_samples, 1))

    return X_list, y_list, y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):

    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs
    images = generator.predict(z_input)

    # create class labels
    y = np.zeros((n_samples, 1))

    return images, y


def train(
    g_model,
    d_model,
    c_model,
    gan_model,
    dataset,
    latent_dim,
    n_epochs=25,
    n_batch=100,
    evals=None,
):
    loss_c = []
    loss_r = []
    loss_f = []
    steps = []

    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset)
    print(X_sup.shape, y_sup.shape)

    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)

    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)

    step = 1

    print(
        "n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d"
        % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps)
    )
    # manually enumerate epochs
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # update supervised discriminator (c)
            Xsup_real, ysup_real, _ = generate_real_samples([X_sup, y_sup], n_batch)
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)

            # update unsupervised discriminator (d)
            X_real, _, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, acc_r = d_model.train_on_batch(X_real, y_real)

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, acc_f = d_model.train_on_batch(X_fake, y_fake)

            # update generator (g)
            X_gan, y_gan = generate_latent_points(latent_dim, n_batch, 4), np.ones(
                (n_batch, 1)
            )
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print("GAN Loss: %.3f" % g_loss)
            print("D_Loss 1: %.3f" % d_loss1, "\n", "D_Loss 2: %.3f" % d_loss2)
            print("D_Acc 1: %.3f" % acc_r, "\n", "D_Acc 2: %.3f" % acc_f)
            # summarize loss on this batch

            loss_c.append(c_loss)
            loss_r.append(d_loss1)
            loss_f.append(d_loss2)
            steps.append(step + 1)

        va_loss, acc = c_model.evaluate(evals[0], y=evals[1], verbose=0)
        print("Model Accuracy: %.3f\nModel Loss %.3f:" % (acc, va_loss))

    return pd.DataFrame(
        data=[steps, loss_c, loss_r, loss_f],
        index=["Step", "Classification Loss", "Real Loss", "Fake Loss"],
    ).transpose()


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches

    def get_config(self):
        return {"patch_size": self.patch_size, "num_patches": self.num_patches}


class AttnBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(AttnBlock, self).__init__()

        self.embedding_dim = embedding_dim

        # Attention
        self.normalize1 = GroupNormalization(groups=4, epsilon=1e-6)

        self.attn = tf.keras.layers.MultiHeadAttention(4, self.embedding_dim)

        # Learned Parameters
        self.normalize2 = GroupNormalization(groups=4, epsilon=1e-6)

        self.D1 = tf.keras.layers.Dense(
            self.embedding_dim, activation=mish, kernel_initializer="he_uniform"
        )
        self.DR1 = tf.keras.layers.Dropout(0.1)
        self.D2 = tf.keras.layers.Dense(
            self.embedding_dim, activation=mish, kernel_initializer="he_uniform"
        )
        self.DR2 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs):

        x_n = self.normalize1(inputs)

        x_attn = self.attn(x_n, x_n)

        x_skip = x_attn + inputs

        x_n = self.normalize2(x_skip)

        x = self.D1(x_n)
        x = self.DR1(x)
        x = self.D2(x)
        x = self.DR2(x)
        x = x_skip + x

        return x

    def get_config(self):
        return {"embedding_dim": self.embedding_dim}


class FNETBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(FNETBlock, self).__init__()

        self.embedding_dim = embedding_dim

        self.N1 = GroupNormalization(groups=4, epsilon=1e-6)

        # Learned Parameters
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.embedding_dim, activation=mish, kernel_initializer="he_uniform"
                ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    self.embedding_dim, activation=mish, kernel_initializer="he_uniform"
                ),
            ]
        )

    def call(self, inputs):

        x = tf.cast(
            tf.signal.fft2d(tf.cast(inputs, dtype=tf.dtypes.complex64)),
            dtype=tf.dtypes.float16,
        )

        x = x + inputs

        x = self.N1(x)

        x_ffn = self.ffn(x)

        x = x + x_ffn

        return x

    def get_config(self):
        return {"embedding_dim": self.embedding_dim}


class GAMixBlock(tf.keras.layers.Layer):
    def __init__(self, h1_sz=288, h2_sz=256):
        super(GAMixBlock, self).__init__()

        self.h1_sz = h1_sz
        self.h2_sz = h2_sz

        self.N = GroupNormalization(groups=4, epsilon=1e-6)

        self.GA_S = tf.keras.layers.GlobalAveragePooling1D()
        self.GA_C = tf.keras.layers.GlobalAveragePooling1D()

        self.C = tf.keras.layers.Concatenate()
        self.D1 = tf.keras.layers.Dense(
            self.h1_sz, activation=mish, kernel_initializer="he_uniform"
        )
        self.D2 = tf.keras.layers.Dropout(0.1)
        self.D3 = tf.keras.layers.Dense(
            self.h2_sz, activation=mish, kernel_initializer="he_uniform"
        )
        self.D4 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs):

        x_in = self.N(inputs)

        x_t = tf.linalg.matrix_transpose(x_in)
        x_1 = self.GA_C(x_t)

        x_2 = self.GA_S(x_in)

        x_3 = self.C([x_1, x_2])

        x = self.D1(x_3)
        x = self.D2(x)
        x = self.D3(x)
        x = self.D4(x)

        return x

    def get_config(self):
        return {"h1_sz": self.h1_sz, "h2_sz": self.h2_sz}


class EmbBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super(EmbBlock, self).__init__()

        self.emb_dim = emb_dim

        self.D1 = tf.keras.layers.Dense(
            self.emb_dim, activation=mish, kernel_initializer="he_uniform"
        )
        self.N = GroupNormalization(groups=4, epsilon=1e-6)
        self.D2 = tf.keras.layers.Dense(
            self.emb_dim, activation=mish, kernel_initializer="he_uniform"
        )
        self.D3 = tf.keras.layers.Dense(
            self.emb_dim, activation=mish, kernel_initializer="he_uniform"
        )
        self.G = tf.keras.layers.GaussianNoise(1e-3)

    def call(self, inputs):

        x = self.D1(inputs)
        x = self.N(x)
        x = self.D2(x)
        x = self.D3(x)
        x = self.G(x)

        return x

    def get_config(self):
        return {"emb_dim": self.emb_dim}


class SignatureAttnNet:
    def __init__(
        self,
        D,
        n_est=3,
        epochs=15,
        batch_size=16,
        patch_sz=4,
        emb_dim=96,
        latent_dim=160,
        use_pos_emb=False,
        optimizer="adamw",
        train_fraction=0.85,
        get_imp=False,
        verbose=2,
    ):

        """
        D: The depth of the FCGR
        n_est: The number of estimators
        epochs: The number of training epochs
        batch_size: The number of batches per epoch
        patch_sz: The size of each super pixle
        emb_dim: The size the embedding dimension for each super pixle
        latent_dim: The size of the latent space for the GAN
        use_pos_emb: Specifies if positional embeddings should be used
        optimizer: Currently only using AdamW
        train_fraction: The proportion of the dataset used to train each estimator
        get_imp: Specifies if Saliency Maps will be computed
        """

        self.D = D
        self.n_est = n_est
        self.epochs = epochs
        self.batch_size = batch_size
        self.patch_sz = patch_sz
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.use_pos_emb = use_pos_emb
        self.optimizer = optimizer
        self.train_fraction = train_fraction
        self.get_imp = get_imp
        self.verbose = verbose

    def fit(self, X, y):
        def define_discriminator(D, n_classes, patch_sz, emb_dim, use_pos_emb):

            # Input
            IN = tf.keras.Input(shape=(2**D, 2**D, 1))

            ####################################################################
            # Make Patches and Embeddings For Attention Block
            n_patches = (2**D // patch_sz) ** 2

            # Add noise to inputs via dropout
            D_A = tf.keras.layers.Dropout(0.125)(IN)

            P_A = Patches(patch_sz, n_patches)(D_A)

            # Get the initial projection
            ATTN_P_ENC = EmbBlock(emb_dim)(P_A)

            # Add positional embeddings
            if use_pos_emb == True:
                img_positions_attn = tf.range(start=0, limit=n_patches, delta=1)
                pos_emb_attn = tf.keras.layers.Embedding(
                    input_dim=n_patches, output_dim=emb_dim
                )(img_positions_attn)

                ATTN_P_ENC = ATTN_P_ENC + pos_emb_attn

            ATTN_P_ENC = AttnBlock(emb_dim)(ATTN_P_ENC)
            ATTN_P_ENC = AttnBlock(emb_dim)(ATTN_P_ENC)
            ATTN_P_ENC = GAMixBlock()(ATTN_P_ENC)
            ####################################################################
            ####################################################################
            # Make Patches and Embeddings For FNET Block
            n_patches = (2**D // patch_sz) ** 2

            # Add noise to inputs via dropout
            D_F = tf.keras.layers.Dropout(0.125)(IN)

            P_F = Patches(patch_sz, n_patches)(D_F)

            # Get the initial projection
            FNET_P_ENC = EmbBlock(emb_dim)(P_F)

            # Add positional embeddings
            if use_pos_emb == True:
                img_positions_fnet = tf.range(start=0, limit=n_patches, delta=1)
                pos_emb_fnet = tf.keras.layers.Embedding(
                    input_dim=n_patches, output_dim=emb_dim
                )(img_positions_fnet)

                FNET_P_ENC = FNET_P_ENC + pos_emb_fnet

            FNET_P_ENC = FNETBlock(emb_dim)(FNET_P_ENC)
            FNET_P_ENC = GAMixBlock()(FNET_P_ENC)
            ####################################################################

            L = tf.keras.layers.Concatenate()([FNET_P_ENC, ATTN_P_ENC])
            L = tf.keras.layers.Dropout(0.375, name="DR_2")(L)
            L = tf.keras.layers.Dense(
                256, activation=mish, kernel_initializer="he_uniform", name="D_1"
            )(L)
            L = tf.keras.layers.Dropout(0.4, name="DR_3")(L)
            L = tf.keras.layers.Dense(n_classes, name="LOGITS")(L)

            # Supervised Output
            OUT_S = tf.keras.layers.Activation("softmax", dtype="float32", name="SMAX")(
                L
            )

            # Compile supervised discriminator model
            OPT_S = Lookahead(
                AdamW(weight_decay=1e-6, beta_1=0.5),
            )

            OPT_S.get_gradients = centralized_gradients_for_optimizer(OPT_S)

            c_model = tf.keras.Model(IN, OUT_S)
            c_model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=OPT_S,
                metrics=["accuracy"],
            )

            logit_model = tf.keras.Model(IN, L)

            # Compile Unsupervised Model
            OUT_U = tf.keras.layers.Lambda(custom_activation, dtype="float32")(L)

            OPT_U = Lookahead(
                AdamW(weight_decay=1e-6, beta_1=0.5),
            )

            OPT_U.get_gradients = centralized_gradients_for_optimizer(OPT_U)

            d_model = tf.keras.Model(IN, OUT_U)
            d_model.compile(
                loss="binary_crossentropy", optimizer=OPT_U, metrics=["accuracy"]
            )

            return c_model, d_model, logit_model

        def generator(latent_dim=self.latent_dim, n_classes=4):

            # Input a random vector
            in_lat = tf.keras.Input(shape=(latent_dim,))

            # Project and reshape
            n_nodes = 8 * 8 * 256
            gen = tf.keras.layers.Dense(
                n_nodes, activation=mish, kernel_initializer="random_normal"
            )(in_lat)
            gen = tf.keras.layers.Dropout(0.25)(gen)
            gen = tf.keras.layers.Reshape((8, 8, 256))(gen)

            # upsample to 16x16
            gen = tf.keras.layers.Conv2DTranspose(
                192,
                (2, 2),
                strides=(2, 2),
                padding="same",
                kernel_initializer="random_normal",
            )(gen)
            gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)

            gen = tf.keras.layers.Conv2DTranspose(
                192, (2, 2), padding="same", kernel_initializer="random_normal"
            )(gen)
            gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)

            # upsample to 32x32
            gen = tf.keras.layers.Conv2DTranspose(
                192,
                (2, 2),
                strides=(2, 2),
                padding="same",
                kernel_initializer="random_normal",
            )(gen)
            gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)

            gen = tf.keras.layers.Conv2DTranspose(
                192, (2, 2), padding="same", kernel_initializer="random_normal"
            )(gen)
            gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)

            # upsample to 64x64
            gen = tf.keras.layers.Conv2DTranspose(
                128,
                (2, 2),
                strides=(2, 2),
                padding="same",
                kernel_initializer="random_normal",
            )(gen)
            gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)

            # Average filters
            out = tf.keras.layers.Conv2D(
                1,
                (1, 1),
                kernel_initializer="random_normal",
            )(gen)
            out = tf.keras.layers.ReLU()(out)

            # define model
            model = tf.keras.Model(in_lat, out)

            return model

        def define_gan(g_model, d_model, c_model):
            # make weights in the discriminator and classifier not trainable
            d_model.trainable = False

            # get noise from generator
            gen_noise = g_model.input

            # get image output from the generator model
            gen_output = g_model.output

            # connect image output from generator as input to discriminator
            gan_output = d_model(gen_output)

            # define gan model as taking noise and outputting score and classifier result
            model = tf.keras.Model(gen_noise, gan_output)

            # compile model
            opt = Lookahead(
                AdamW(weight_decay=1e-6, beta_1=0.5),
            )

            opt.get_gradients = centralized_gradients_for_optimizer(opt)

            model.compile(
                loss=[
                    "binary_crossentropy",
                ],
                optimizer=opt,
            )

            return model

        # Encode y and prepare variables for holding statistics and importance scores
        self.encoder = OneHotEncoder(sparse=False).fit(y.reshape(-1, 1))
        self.label_encodings = LabelEncoder().fit(y)
        self.y_shape = self.encoder.categories_[0].shape[0]

        # Prepare variables
        self.model_list = []
        self.feature_importances_ = {}
        self.scores_ = []
        self.feature_importance_order_ = self.encoder.categories_[0]
        self.history = []

        for iter in range(self.n_est):

            if self.verbose > 0:
                print("ITERATION:", iter + 1)

            tf.keras.backend.clear_session()

            # Generate test-train data and oversample
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=self.train_fraction,
                shuffle=True,
                stratify=y,
            )

            y_unique, y_counts = np.unique(y_train, return_counts=True)

            min_samples = 50

            X_re = []
            y_re = []
            y_dict = {}
            for i, y_un in enumerate(y_unique):
                if y_counts[i] > min_samples:
                    y_dict[y_un] = y_counts[i]
                else:
                    y_dict[y_un] = min_samples

            X_re, y_re = RandomOverSampler(
                sampling_strategy=y_dict,
            ).fit_resample(np.asarray([x.flatten() for x in X_train]), y_train)

            X_re = np.asarray([x.reshape(2**self.D, 2**self.D) for x in X_re])

            y_re = self.label_encodings.transform(y_re).astype(int)

            # Prepare models
            c_model, d_model, l_model = define_discriminator(
                self.D, self.y_shape, self.patch_sz, self.emb_dim, self.use_pos_emb
            )

            g_model = generator(self.latent_dim, self.y_shape)

            gan_model = define_gan(g_model, d_model, c_model)

            if self.verbose > 1:
                c_model.summary()
                d_model.summary()
                g_model.summary()
                gan_model.summary()

            # Create and train model
            hist = train(
                g_model,
                d_model,
                c_model,
                gan_model,
                (X_re, y_re),
                self.latent_dim,
                n_epochs=self.epochs,
                n_batch=self.batch_size,
                evals=(X_test, self.label_encodings.transform(y_test)),
            )

            self.history.append(hist)

            # Save weights
            self.model_list.append((c_model.get_weights(), c_model.get_config()))

            # Get feature importances
            if self.get_imp:
                X_sal = X_test[..., np.newaxis]

                self.calc_saliency(X_sal, y_test, l_model)

            del c_model
            del d_model
            del g_model
            del l_model
            del gan_model
            collect()

        return self

    def calc_saliency(self, X, y, logit_model, smooth=30, noise=0.2):

        tmp_dict = {}

        for class_name in self.feature_importance_order_:

            output_loc = np.argmax(
                np.where(self.feature_importance_order_ == class_name, 1, 0)
            )

            tax_loc = np.where(y == class_name, True, False)

            img_titles = y[tax_loc]

            X_loc = X[tax_loc]

            if len(img_titles) > 0:

                saliency = Saliency(logit_model, clone=True)

                mapping = saliency(
                    CategoricalScore([output_loc]),
                    X_loc,
                    smooth_samples=smooth,
                    smooth_noise=noise,
                )

                if class_name not in tmp_dict:
                    tmp_dict[class_name] = []

                [tmp_dict[class_name].append(entry.flatten()) for entry in mapping]

        for key, maps in tmp_dict.items():
            mean_maps = np.mean(maps, axis=0)

            if key not in self.feature_importances_:
                self.feature_importances_[key] = []

            self.feature_importances_[key].append(mean_maps)

        del saliency
        collect()

    def get_maps(self):
        tax_maps = []

        for taxa, maps in self.feature_importances_.items():

            if len(maps) > 1:
                tmp_map = np.mean(maps, axis=0)
                tmp_map = tmp_map.reshape(2**self.D, 2**self.D)
                tax_maps.append(tmp_map)

            else:
                tmp_map = maps[0].reshape(2**self.D, 2**self.D)
                tax_maps.append(tmp_map)

        return tax_maps

    def display_maps(self, fname="saliency_map.svg"):

        figure, ax = plt.subplots(
            nrows=1, ncols=len(self.feature_importances_), figsize=(12, 4)
        )

        i = 0
        for taxa, maps in self.feature_importances_.items():
            ax[i].set_title(taxa, fontsize=14)

            if len(maps) > 1:
                tmp_map = np.mean(maps, axis=0)
                tmp_map = tmp_map.reshape(2**self.D, 2**self.D)
                ax[i].imshow(tmp_map, cmap="jet")

            else:
                tmp_map = maps[0].reshape(2**self.D, 2**self.D)
                ax[i].imshow(tmp_map, cmap="jet")

            ax[i].axis("off")

            i += 1

        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def predict_proba(self, X):

        predictions = np.zeros((X.shape[0], self.y_shape), dtype=float)

        for model in self.model_list:

            tf.keras.backend.clear_session()

            tmp_model = tf.keras.Model.from_config(
                model[1],
                custom_objects={
                    "Patches": Patches,
                    "AttnBlock": AttnBlock,
                    "FNETBlock": FNETBlock,
                    "GAMixBlock": GAMixBlock,
                    "EmbBlock": EmbBlock,
                },
            )

            tmp_model.set_weights(model[0])

            tmp_model.trainable = False

            predictions += tmp_model.predict(X)

        return predictions / float(self.n_est)

    def predict(self, X):

        p = self.predict_proba(X)

        p = np.argmax(p, axis=1)

        return self.label_encodings.inverse_transform(p)


################################################################################################
