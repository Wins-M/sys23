import warnings
warnings.simplefilter("ignore")

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn

# seaborn.set_style("darkgrid")
seaborn.set_style("white")

plt.rc("figure", figsize=(16, 6))
# plt.rc("figure", figsize=(6, 3))

plt.rc("savefig", dpi=90)

# plt.rc("font", family="sans-serif")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rc("font", size=12)
# plt.rc("font", size=10)

plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"
