import pandas as pd
import matplotlib.pyplot as plt
d = pd.read_pickle("./scores_untouched.pkl")

plt.bar(d)
plt.show()

