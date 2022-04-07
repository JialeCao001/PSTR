import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(nrows=1, ncols=2)

# x = np.arange(1,7)
x = ['50','100','500','1000','2000','4000']
ax1 = plt.subplot(121)

Ours = [95.90, 95.27, 93.24, 92.20, 90.71, 88.75]
ax1.plot(x, Ours, '--sr', label='PSTR-PVT')
Ours = [94.30,  93.55, 91.34,  89.55, 87.86, 85.34]
ax1.plot(x, Ours, '-sr', label='PSTR-R50')
TCTS = [94.46, 93.87, 90.70, 89.17, 86.84, 84.26]
ax1.plot(x, TCTS, '-ob', label='TCTS')
RDLR = [93.80, 92.99, 89.55, 87.46, 84.97, 82.67]
ax1.plot(x, RDLR, '-vg', label='RDLR')
CLSA = [88.42, 87.30, 85.42, 84.62, 83.35, 77.51]
ax1.plot(x, CLSA, '-pc', label='CLSA')
MGTS = [84.70, 82.95, 76.88, 73.99, 70.34, 66.47]
ax1.plot(x, MGTS, '-dm', label='MGTS')
ax1.set_ylabel('mAP')
ax1.set_xlabel('Gallery Size')
ax1.set_ylim([64, 96])
ax1.legend(loc='lower left')


ax2 = plt.subplot(122)
Ours = [95.90, 95.27, 93.24, 92.20, 90.71, 88.75]
ax2.plot(x, Ours, '--sr', label='PSTR-PVT')
Ours = [94.30,  93.55, 91.34,  89.55, 87.86, 85.34]
ax2.plot(x, Ours, '-sr', label='PSTR-R50')
AlignPS = [94.70, 94.06, 90.86, 89.11, 86.78, 84.03]
ax2.plot(x, AlignPS, '-ob', label='AlignPS+')
DKD = [93.98, 93.14, 89.96, 88.28, 86.08, 83.63]
ax2.plot(x, DKD, '-vg', label='DKD')
NAE = [93.10, 92.10, 87.13, 84.83, 81.96, 79.00]
ax2.plot(x, NAE, '-pc', label='NAE+')
CTXG = [87.05, 84.06, 78.51, 74.68, 70.99, 65.64]
ax2.plot(x, CTXG, '-dm', label='CTXG')
RCAA = [83.75, 79.33, 70.76, 64.15, 60.96, 55.94]
ax2.plot(x, RCAA, '-hy', label='RCAA')
OIM = [79.34, 75.45, 65.66, 60.85, 56.46, 51.27]
ax2.plot(x, OIM, '->k', label='OIM', color='purple')
ax2.set_ylabel('mAP')
ax2.set_xlabel('Gallery Size')
ax2.set_ylim([48, 96])
ax2.legend(loc='lower left')


fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
plt.savefig('./cuhk.jpg')