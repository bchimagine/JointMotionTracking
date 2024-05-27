import numpy as np
import matplotlib.pyplot as plt

# Load data from the first .npy file
data1 = np.load('trans_error_withoutdef.npy')

# Load data from the second .npy file
data2 = np.load('angular_error_withoutdef.npy')

# Create a list of data to be plotted
data_to_plot = [data1, data2]
mean1 = np.mean(data1)
variance1 = np.var(data1)
mean2 = np.mean(data2)
variance2 = np.var(data2)

print(f"Data 1 Mean: {mean1}, Variance: {variance1}")
print(f"Data 2 Mean: {mean2}, Variance: {variance2}")
# Create a box and whisker plot
plt.boxplot(data_to_plot)

# Set labels for the boxplot
plt.xticks([1, 2], ['Data 1', 'Data 2'])
plt.ylabel('Values')

# Set a title for the plot
plt.title('Box and Whisker Plot')

# Show the plot
plt.show()
