import re

import matplotlib.pyplot as plt

# Initialize lists to store epochs, train losses and validation losses
epochs = []
train_loss = []
val_loss = []

# Open the log file
prev_epoch = 0

with open("train.log", "r") as f:
    for line in f:
        # Find lines with epoch info
        if "info" in line:
            # Extract epoch number
            epoch = int(re.search(r"Epoch (\d+)", line).group(1))
            if epoch != prev_epoch:
                print(prev_epoch, epoch)
                # Extract loss values
                # pattern = r'loss (.*?)\n'
                pattern = r"[-+]?\d*\.\d+"
                loss = float(re.search(pattern, line).group())
                if "Train" in line:
                    epochs.append(epoch)
                    train_loss.append(loss)
                elif "Val" in line:
                    val_loss.append(loss)
                    prev_epoch = epoch

# Create the plot
plt.figure(figsize=(10, 5))

# Plot training and validation loss
plt.plot(epochs, train_loss, label="Training Loss", color="blue")
plt.plot(epochs, val_loss, label="Validation Loss", color="red")

# Add horizontal lines at the minimum values
plt.axhline(
    min(train_loss), color="blue", linestyle="--", label="Min Training Loss"
)
plt.axhline(
    min(val_loss), color="red", linestyle="--", label="Min Validation Loss"
)

# Annotate the minimum values on the y-axis
plt.text(
    0,
    min(train_loss),
    "{:.2f}".format(min(train_loss)),
    va="center",
    ha="left",
    backgroundcolor="w",
)
plt.text(
    0,
    min(val_loss),
    "{:.2f}".format(min(val_loss)),
    va="center",
    ha="left",
    backgroundcolor="w",
)

# Add legend, title, and x, y labels
plt.legend(loc="upper right")
plt.title("Training and Validation Loss Over Epochs")
plt.ylabel("Loss Value")
plt.xlabel("Epochs")

# Save the plot as a .png file
plt.savefig("train_val_loss.png")

# Show the plot
# plt.show()
