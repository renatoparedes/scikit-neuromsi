
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import attr
sns.reset_defaults()

@attr.s
class MisDatos:
    arr0 = attr.ib(repr=False)
    arr1 = attr.ib(repr=False)
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        kwargs.setdefault("size", 10)
        ax = sns.scatterplot(x=self.arr0, y=self.arr1, ax=ax, **kwargs)
        ax.set_title("Zaraza de cosito")
        ax.set_xlabel("Zaraza")
        ax.set_ylabel("Cosito")
        
        return ax
    
# ES SU NOTEBOOK
    
md = MisDatos(
    np.random.normal(size=1000), 
    np.random.normal(size=1000))

fig, ax = plt.subplots(1,2)
fig.set_size_inches(10, 10)
ax = md.plot(ax=ax[0])
plt.tight_layout()
