"""
accauracy: 
48.043676069153776, 52.4416135881104, 48.801941158629056, 48.34698210494388, 50.621777373369724, 51.8956627236882, 53.472854109796785, 53.07855626326964, 53.92781316348195, 54.382772217167116, 55.868971792538666, 55.44434334243251, 55.99029420685472, 61.20715802244465, 62.26872914771005, 64.51319381255686, 63.512283894449496, 66.87898089171973, 67.94055201698514, 68.30451925993327, 68.9111313315135, 71.51956323930845, 73.09675462541705, 72.975432211101, 74.21898695784046, 74.64361540794661, 74.58295420078859, 75.58386411889597, 75.55353351531696, 75.09857446163178

loss: 
11.740347163605941, 8.829523506523469, 5.085878912073869, 3.2873291547680883, 2.322169851860588, 1.8738625077713869, 1.5269737029126325, 1.4394840805810964, 1.434182773798531, 1.2932932289878216, 1.2534780258073073, 1.2688587728527734, 1.241306545132356, 0.937132778541741, 0.8853267268705293, 0.8603156508839079, 0.8613275027028133, 0.7544607297593156, 0.738125630946445, 0.7315617589702545, 0.7186301969197343, 0.6576075921624474, 0.6403682137652609, 0.6280996254118703, 0.6006174283373661, 0.5917234915453385, 0.5874717859329187, 0.5815756053687812, 0.5774836305951975, 0.5705471087801189

LR reduced from 0.001 to 0.0005 on epoch 14 (seen from increase of acurracy 55.99029420685472 to 61.20715802244465)
LR reduced from 0.0005 to 0.00025 on epoch 18
LR reduced from 0.00025 to 0.000125 on epoch 22
LR reduced from 0.000125 to 0.0000625 on epoch 24 (72.975432211101 to 74.21898695784046)


test accauracy:
72.3823975720789

test loss:
0.7442537257896777
"""
import matplotlib.pyplot as plt


accuracy = [48.043676069153776, 52.4416135881104, 48.801941158629056, 48.34698210494388,
            50.621777373369724, 51.8956627236882, 53.472854109796785, 53.07855626326964,
            53.92781316348195, 54.382772217167116, 55.868971792538666, 55.44434334243251,
            55.99029420685472, 61.20715802244465, 62.26872914771005, 64.51319381255686,
            63.512283894449496, 66.87898089171973, 67.94055201698514, 68.30451925993327,
            68.9111313315135, 71.51956323930845, 73.09675462541705, 72.975432211101,
            74.21898695784046, 74.64361540794661, 74.58295420078859, 75.58386411889597,
            75.55353351531696, 75.09857446163178]

loss = [11.740347163605941, 8.829523506523469, 5.085878912073869, 3.2873291547680883,
        2.322169851860588, 1.8738625077713869, 1.5269737029126325, 1.4394840805810964,
        1.434182773798531, 1.2932932289878216, 1.2534780258073073, 1.2688587728527734,
        1.241306545132356, 0.937132778541741, 0.8853267268705293, 0.8603156508839079,
        0.8613275027028133, 0.7544607297593156, 0.738125630946445, 0.7315617589702545,
        0.7186301969197343, 0.6576075921624474, 0.6403682137652609, 0.6280996254118703,
        0.6006174283373661, 0.5917234915453385, 0.5874717859329187, 0.5815756053687812,
        0.5774836305951975, 0.5705471087801189]

testAccuracy = 72.3823975720789
testLoss = 0.7442537257896777

epochs = list(range(1, 31))
lrDrops = {14: "→0.0005", 18: "→0.00025", 22: "→0.000125", 25: "→0.0000625"}

fig, ax1 = plt.subplots(figsize=(10,5))

# Training Accuracy
ax1.plot(epochs, accuracy, marker="o", markersize=4, linewidth=1.5, color="tab:blue", label="Accuracy (%)")
y_max_acc = 80
y_min_acc = 45
ax1.set_ylim([y_min_acc, y_max_acc])
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Accuracy (%)", color="tab:blue", fontsize=11)
ax1.tick_params(axis='y', labelcolor="tab:blue")

# LR annotations for accuracy
for e, txt in lrDrops.items():
    if e <= len(accuracy):
        y_val = accuracy[e-1]
        ax1.vlines(e, ymin=y_min_acc, ymax=y_val, linestyle="--", color="gray", linewidth=1)
        y_text = min(y_val + (y_max_acc - y_min_acc) * 0.03, y_max_acc)
        ax1.text(e, y_text, f"LR {txt}", rotation=45, fontsize=7, color="gray", ha='left', va='bottom')

# Training Loss on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(epochs, loss, marker="o", markersize=4, linewidth=1.5, color="tab:red", label="Loss")
y_max_loss = max(loss)
y_min_loss = 0
ax2.set_ylim([y_min_loss, y_max_loss * 1.05])
ax2.set_ylabel("Loss", color="tab:red", fontsize=11)
ax2.tick_params(axis='y', labelcolor="tab:red")

# LR annotations for loss
for e, txt in lrDrops.items():
    if e <= len(loss):
        y_val = loss[e-1]
        ax2.vlines(e, ymin=y_min_loss, ymax=y_val, linestyle="--", color="gray", linewidth=1)
        y_text = min(y_val + (y_max_loss - y_min_loss) * 0.03, y_max_loss)
        ax2.text(e, y_text, f"LR {txt}", rotation=45, fontsize=7, color="gray", ha='left', va='bottom')

ax1.set_title("Training Curves (Accuracy & Loss)", fontsize=13)
fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(8,4))

# Test Accuracy
axes[0].bar(["Test Accuracy"], [testAccuracy], color="tab:orange")
axes[0].set_ylim([0, 100])
axes[0].set_ylabel("Accuracy (%)", fontsize=11)
axes[0].set_title("Final Test Accuracy", fontsize=12)

# Test Loss
axes[1].bar(["Test Loss"], [testLoss], color="tab:green")
axes[1].set_ylim([0, max(1, testLoss*1.5)])
axes[1].set_ylabel("Loss", fontsize=11)
axes[1].set_title("Final Test Loss", fontsize=12)

plt.tight_layout()
plt.show()