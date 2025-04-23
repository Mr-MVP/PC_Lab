import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv("execution_times.csv")

# Plot the execution times
plt.figure(figsize=(10, 5))
plt.plot(data["N"], data["Serial Time"], label="Serial Execution Time", color='r')
plt.plot(data["N"], data["Parallel Time"], label="Parallel Execution Time", color='b')

# Labels and title
plt.xlabel("Matrix Size (N)")
plt.ylabel("Execution Time (seconds)")
plt.title("Serial vs Parallel Execution Time for Matrix Multiplication")
plt.legend()
plt.grid()

# Show plot
plt.show()
plt.savefig("execution_times.png",dpi=300)