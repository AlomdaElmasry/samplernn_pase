import matplotlib.pyplot as plt

epochs = [10, 20, 30, 40, 50]
pase_durs = [1, 10, 60, 120]

# Loss Validation Plot

emb_ling_loss = [2.215, 2.138, 2.106, 2.079, 2.067]
pase_ling_loss = [2.204, 2.127, 2.093, 2.066, 2.055]
emb_lf0_loss = [2.213, 2.136, 2.102, 2.074, 2.059]
pase_lf0_loss = [2.203, 2.124, 2.087, 2.066, 2.039]

plt.xlabel('Epoch (n)')
plt.ylabel('Loss')
plt.plot(epochs, emb_ling_loss)
plt.plot(epochs, emb_lf0_loss)
plt.plot(epochs, pase_ling_loss)
plt.plot(epochs, pase_lf0_loss)
plt.legend(['Embedding + Linguistic', 'Embedding + Linguistic + LF0', 'PASE + Linguistic', 'PASE + Linguistic + '
                                                                                           'LF0'], loc='upper left')
plt.show()

# MCD Plot

emb_ling_mcd = [11.66, 11.01, 10.78, 10.23, 11.46]
emb_lf0_mcd = [11.17, 10.81, 10.32, 10.03, 10.36]
pase_ling_mcd = [11.40, 10.86, 10.56, 10.57, 10.55]
pase_lf0_mcd = [11.98, 10.58, 11.40, 10.47, 10.07]

plt.xlabel('Epoch (n)')
plt.ylabel('MCD (dB)')
plt.plot(epochs, emb_ling_mcd)
plt.plot(epochs, emb_lf0_mcd)
plt.plot(epochs, pase_ling_mcd)
plt.plot(epochs, pase_lf0_mcd)
plt.legend(['Embedding + Linguistic', 'Embedding + Linguistic + LF0', 'PASE + Linguistic', 'PASE + Linguistic + '
                                                                                           'LF0'], loc='upper left')
plt.show()

# F0 RMSE

emb_ling_f0 = [52.48, 40.35, 42.42, 35.77, 33.81]
pase_ling_f0 = [42.74, 38.69, 37.42, 38.20, 33.88]
emb_lf0_f0 = [36.10, 29.12, 28.76, 23.47, 20.55]
pase_lf0_f0 = [28.84, 22.00, 28.86, 22.53, 20.27]

plt.xlabel('Epoch (n)')
plt.ylabel('F0 RMSE (Hz)')
plt.plot(epochs, emb_ling_f0)
plt.plot(epochs, emb_lf0_f0)
plt.plot(epochs, pase_ling_f0)
plt.plot(epochs, pase_lf0_f0)
plt.legend(['Embedding + Linguistic', 'Embedding + Linguistic + LF0', 'PASE + Linguistic', 'PASE + Linguistic + '
                                                                                           'LF0'], loc='upper left')
plt.show()
#
# # Accuracy
#
# emb_ling_acc = [0.778, 0.831, 0.836, 0.839, 0.884]
# pase_ling_acc = [0.781, 0.829, 0.834, 0.828, 0.841]
# emb_lf0_acc = [0.857, 0.88, 0.881, 0.889, 0.891]
# pase_lf0_acc = [0.857, 0.885, 0.864, 0.888, 0.889]
#
# plt.xlabel('Epoch')
# plt.ylabel('V/UV Accuracy')
# plt.plot(epochs, emb_ling_acc)
# plt.plot(epochs, emb_lf0_acc)
# plt.plot(epochs, pase_ling_acc)
# plt.plot(epochs, pase_lf0_acc)
# plt.legend(['Embedding + Linguistic', 'Embedding + Linguistic + LF0', 'PASE + Linguistic', 'PASE + Linguistic + '
#                                                                                            'LF0'], loc='upper left')
# plt.show()


# PASE Duration
ling_mcd = [12.95, 11.56, 11.31, 11.23]
ling_rmse = [46.96, 41.84, 38.43, 39.35]
lf0_mcd = []
lf0_rmse = []

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('PASE Chunk Duration (s)')
ax1.set_ylabel('F0 RMSE (Hz)')
ax2.set_ylabel('MCD (dB)')
ax1.plot(pase_durs, ling_rmse, color='black')
ax2.plot(pase_durs, ling_mcd, color='blue')
ax1.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='blue')

plt.show()
