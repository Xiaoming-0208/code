import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mean = 20.22
vari = 0.96


def target_distribution(x):
    return np.exp(-(x - mean) ** 2.0 / (2.0 * vari ** 2.0)) / (np.sqrt(2.0 * np.pi) * vari)


def proposal_distribution(x, step):
    return np.random.uniform(x - step, x + step)


def met_has(target_distribution, proposal_distribution, step, num_samples, x_ini):
    samples = np.zeros(num_samples)
    x = x_ini
    for i in range(num_samples):
        x_proposal = proposal_distribution(x, step)
        acceptance_ratio = target_distribution(x_proposal) / target_distribution(x)
        if np.random.uniform(0, 1) < acceptance_ratio:
            x = x_proposal
        samples[i] = x
    return samples


step = 0.1
num_samples = 60000
x_ini = int(mean)

samples = met_has(target_distribution, proposal_distribution, step, num_samples, x_ini)

num_bins = 100
plt.hist(samples,
         num_bins,
         density=1,
         facecolor='red',
         alpha=0.6,
         label='Sample Distribution')
plt.legend()
plt.show()

dataframe = pd.DataFrame(samples)
#dataframe.to_csv('K_5_100.csv')
