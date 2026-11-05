import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.cm as cm


def create_imbalanced_subset(x, y, alpha, dataset_length, min_class_samples=50):
    """
    Creates an imbalanced subset of the dataset based on a Dirichlet distribution.

    Parameters:
    - x: Input data (numpy array).
    - y: One-hot encoded labels (numpy array).
    - alpha: Dirichlet alpha parameter controlling the imbalance.
    - dataset_length: Desired length of the imbalanced subset.
    - min_class_samples: Minimum number of samples per class.

    Returns:
    - x_imbalanced: Imbalanced subset of input data.
    - y_imbalanced: Imbalanced subset of labels.
    """
    # Calculate the initial class distribution
    class_counts = np.sum(y, axis=0)
    num_classes = len(class_counts)

    # Create a probability distribution over the classes
    class_probabilities = np.random.dirichlet(alpha * np.ones(num_classes))

    # Create a list of indices for each class
    class_indices = [np.where(y[:, i] == 1)[0] for i in range(num_classes)]

    # Flatten and shuffle indices
    all_indices = np.concatenate(class_indices)
    np.random.shuffle(all_indices)

    # Remove instances until the desired dataset length is reached
    while len(all_indices) > dataset_length:
        sampled_class = np.random.choice(num_classes, p=class_probabilities)

        # Skip if the sampled class is exhausted
        if len(class_indices[sampled_class]) == 0:
            continue

        # Remove a random instance of the sampled class
        remove_index = np.random.choice(class_indices[sampled_class])
        all_indices = all_indices[all_indices != remove_index]
        class_indices[sampled_class] = class_indices[sampled_class][
            class_indices[sampled_class] != remove_index
        ]

        # Adjust class probabilities if a class has fewer than the minimum samples
        if len(class_indices[sampled_class]) < min_class_samples:
            class_probabilities[sampled_class] = 0
            class_probabilities = class_probabilities / np.sum(class_probabilities)

    # Create the imbalanced dataset
    x_imbalanced = x[all_indices]
    y_imbalanced = y[all_indices]
    return x_imbalanced, y_imbalanced


def plot_label_distribution(y):
    """
    Plots the label distribution of a dataset.

    Parameters:
    - y: Labels (numpy array, one-hot or class indices).
    """
    if len(y.shape) > 1:  # Convert one-hot to class indices if needed
        y = np.argmax(y, axis=1)

    num_classes = len(np.unique(y))
    plt.bar(
        np.arange(num_classes),
        np.bincount(y),
        align="center",
        color=plt.cm.tab10.colors,
    )
    plt.xticks(np.arange(num_classes))
    plt.title("Klassenverteilung")
    plt.xlabel("Klasse")
    plt.ylabel("Häufigkeit")
    plt.show()

def gradcam_analysis(
    img,
    model,
    last_conv_layer_name,
    pred_class,
    true_class,
    pred_prob,
    label_names,
    alpha=0.6,
    title=None,
):
    """
    Performs Grad-CAM analysis and visualizes the heatmap.

    Parameters:
    - img: Input image (numpy array).
    - model: Trained model.
    - last_conv_layer_name: Name of the last convolutional layer.
    - pred_class: Predicted class index.
    - true_class: True class index.
    - pred_prob: Predicted probability for the class.
    - label_names: List of class names.
    - alpha: Opacity of the heatmap overlay.
    - title: Title for the visualization (optional).
    """
    img_array = tf.image.resize(img, (224, 224))
    img_array = tf.expand_dims(img_array, axis=0)

    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_class]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    img = tf.image.resize(img, (224, 224))
    img = tf.keras.preprocessing.image.array_to_img(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        np.array(jet_heatmap) * alpha + np.array(img)
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    if title:
        fig.suptitle(
            title
            + f"\nVorhergesagt: {label_names[pred_class]}, Wahr: {label_names[true_class]}, Konfidenz: {pred_prob:.2f}"
        )
    else:
        fig.suptitle(
            f"Vorhergesagt: {label_names[pred_class]}, Wahr: {label_names[true_class]}, Konfidenz: {pred_prob:.2f}"
        )

    axs[0].imshow(img)
    axs[0].axis("off")
    axs[1].imshow(superimposed_img)
    axs[1].axis("off")
    plt.show()
    
def plot_metrics(hist):
    # Plot training & validation metrics values
    plt.figure(figsize=(15, 5))

    # Plot accuracy and macro accuracy
    plt.subplot(1, 3, 1)
    plt.plot(hist.history["accuracy"], "r", label="Training: Accuracy")
    plt.plot(hist.history["macro_f1_score"], "b", label="Training: Macro-F1")
    plt.plot(hist.history["val_accuracy"], "r--", label="Test: Accuracy")
    plt.plot(hist.history["val_macro_f1_score"], "b--", label="Test: Macro-F1")
    plt.title("Accuracy und F1-Score")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoche")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.xticks(np.arange(len(hist.history["accuracy"])))
    plt.grid(True)

    # Annotate maximum values under the legends
    max_train_acc = max(hist.history["accuracy"])
    max_val_acc = max(hist.history["val_accuracy"])
    max_train_f1 = max(hist.history["macro_f1_score"])
    max_val_f1 = max(hist.history["val_macro_f1_score"])
    plt.annotate(f'Max Train Acc: {max_train_acc:.2f} | Max Test Acc: {max_val_acc:.2f} \n Max Train F1: {max_train_f1:.2f} | Max Val F1: {max_val_f1:.2f}', 
                 xy=(0.5, -.6), xytext=(0, -10), xycoords='axes fraction', textcoords='offset points', ha='center')

    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(hist.history["loss"], "k", label="Training: Verlust")
    plt.plot(hist.history["val_loss"], "k--", label="Test: Verlust")
    plt.title("Modellverlust")
    plt.ylabel("Loss")
    plt.xlabel("Epoche")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.xticks(np.arange(len(hist.history["loss"])))
    plt.grid(True)

    # Annotate minimum loss values under the legends
    min_train_loss = min(hist.history["loss"])
    min_val_loss = min(hist.history["val_loss"])
    plt.annotate(f'Min Training: Verlust: {min_train_loss:.2f} | Min Val Loss: {min_val_loss:.2f}', 
                 xy=(0.5, -.6), xytext=(0, -10), xycoords='axes fraction', textcoords='offset points', ha='center')

    plt.tight_layout()
    plt.show()
